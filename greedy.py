import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time
import heapq
from warehouse_env import WarehouseEnv
from multi_agent_decentralized import extract_full_state_vector
from maddpg import MultiAgentReplayBuffer
import pickle

# Directions and their respective orientation codes
DIRECTIONS = {
    0: (-1, 0),  # Up
    1: (0, 1),   # Right
    2: (1, 0),   # Down
    3: (0, -1),  # Left
}

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def valid_move(grid, x, y):
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != 1 and grid[x][y] != 2 and grid[x][y] != 3

def assign_goal(obs, agent_id):
    """
    Assign the closest request to the agent if unassigned.
    
    Args:
        obs: Centralized observation dictionary
        agent_id: ID of the agent
        
    Returns:
        tuple: (goal_position, task_type) where task_type is 'pickup', 'delivery', or 'idle'
    """
    requests = obs['requests']
    agent_pos = obs['agents'][agent_id]['position']
    # The carrying attribute is a numpy array with a single value, need to extract it
    carrying = obs['agents'][agent_id]['carrying'][0]
    # print(f"Agent {agent_id} pos: {agent_pos}, carrying: {carrying}")
    
    if carrying == -1:  # Not carrying anything
        # Find closest unpicked request
        # print(f"Looking for closest unpicked request, requests: {requests}")
        min_dist = float('inf')
        min_dist_idx = -1
        for i, req in enumerate(requests):
            # Skip placeholders
            if req['id'] == -1:
                continue
            # Only consider requests that haven't been picked yet
            if not req['picked']:
                src = tuple(req['source'])
                # print(f"Request {i} id: {req['id']}, src: {src}, picked: {req['picked']}")
                dist = heuristic(agent_pos, src)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_idx = i
                # print(f"Min dist: {min_dist}, min dist idx: {min_dist_idx}")

        if min_dist_idx != -1:
            return tuple(requests[min_dist_idx]['source']), 'pickup'
        else:
            return -1, 'idle'
    else:
        # Move to request destination for delivery
        for req in requests:
            if req['id'] == carrying:
                return tuple(req['destination']), 'delivery'
        # If we can't find the request, idle
        return -1, 'idle'
    
def greedy(grid, start, goal, agent_id, obs):
    """
    Implements the D* path planning algorithm for warehouse agents.
    
    Args:
        grid: The warehouse grid
        start: Starting position of the agent
        goal: Goal position
        agent_id: ID of the agent
        obs: Observation dictionary
        
    Returns:
        int: Action to take (0-6)
    """
    if goal == -1:
        return 6  # Idle if no goal
        
    # Get agent orientation
    orientation = obs['agents'][agent_id]['orientation']
    
    # Calculate best direction to goal
    best_dir = None
    min_dist = float('inf')
    
    for dir_idx, (dx, dy) in DIRECTIONS.items():
        # Calculate the next position if we move in this direction
        next_pos = (start[0] + dx, start[1] + dy)
        
        # Check if this is a valid move and would bring us closer to the goal
        if valid_move(grid, next_pos[0], next_pos[1]):
            dist = heuristic(next_pos, goal)
            if dist < min_dist or (dist == min_dist and dir_idx == orientation):
                min_dist = dist
                best_dir = dir_idx
    
    if best_dir is None:
        return 6  # Idle if no valid direction
    
    # Determine action based on current orientation vs best direction
    if orientation == best_dir:
        return 0  # Move forward
    elif (orientation + 1) % 4 == best_dir:
        return 3  # Rotate right
    elif (orientation - 1) % 4 == best_dir:
        return 2  # Rotate left
    else:
        # Need to rotate 180 degrees - choose the shorter rotation
        return 2  # Rotate left (arbitrary choice)

def fill_buffer():
    # Environment setup
    num_agents = 3
    env = WarehouseEnv(
        num_agents=num_agents, 
        grid_size=(15, 9), 
        goal_deliveries=10, 
        poisson_lambda=0.1, 
        max_requests=5, 
        observation_type='decentralized',
        normalize=True
    )
    
    # Setup MADDPG replay buffer
    actor_dims = [env.observation_space.shape[1]] * env.num_agents
    critic_dims = extract_full_state_vector(env, env._get_obs()[1]).shape[0]
    n_actions = env.action_space[0].n
    batch_size = 128
    buffer_size = 10000
    buffer = MultiAgentReplayBuffer(buffer_size, critic_dims, actor_dims, n_actions, num_agents, batch_size)
    
    print(f"Starting D* algorithm to collect data for MADDPG training...")
    print(f"Number of agents: {num_agents}")
    print(f"Actor dimensions: {actor_dims}")
    print(f"Critic dimensions: {critic_dims}")
    print(f"Action space size: {n_actions}")
    
    # Initialize buffer counter
    total_steps = 0
    skipped_steps = 0
    all_episode_rewards = []
    # Continue until buffer is full
    while buffer.mem_cntr < buffer_size:
        # Reset environment and get initial observations
        obs, info = env.reset()
        
        # Get centralized observation for goal assignment
        centralized_obs = env._obs_centralized()
        
        # Initialize episode parameters
        t = 0
        terminated = False
        truncated = False
        max_episode_steps = 500  # Limit on steps per episode
        episode_reward = 0
        
        # Episode loop
        while not (np.any(terminated) or np.any(truncated)) and t < max_episode_steps:
            if total_steps % 1000 == 0:
                print(f"Total steps: {total_steps}, Buffer: {buffer.mem_cntr}/{buffer_size}")
            
            # Store old observations for replay buffer
            obs_old = obs.copy()
            state_old = extract_full_state_vector(env, info)
            
            # Determine actions for all agents
            actions = []
            for agent_id in range(num_agents):
                # Get goal for the agent from centralized observation
                agent_goal, task = assign_goal(centralized_obs, agent_id)
                
                # If agent has no goal, idle
                if agent_goal == -1:
                    action = 6  # Idle action
                else:
                    # Get agent's current position and orientation
                    agent_pos = centralized_obs['agents'][agent_id]['position']
                    orientation = centralized_obs['agents'][agent_id]['orientation']
                    
                    # Calculate the position in front of the agent
                    dx, dy = DIRECTIONS[orientation]
                    front_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
                    
                    # If the target is directly in front of the agent, perform pick/deliver action
                    if front_pos == agent_goal:
                        if task == 'pickup':
                            action = 4  # Pick
                        elif task == 'delivery':
                            action = 5  # Deliver
                        else:
                            action = 6  # Idle
                    else:
                        # Need to navigate to the goal
                        action = greedy(centralized_obs['grid'], agent_pos, agent_goal, agent_id, centralized_obs)
                
                actions.append(action)
            
            # Take step in environment
            obs_new, rewards, terminated, truncated, info_new = env.step(actions)
            episode_reward += sum(rewards) / num_agents
            # Update centralized observation
            centralized_obs = env._obs_centralized()
            
            # Extract full state for critic
            state_new = extract_full_state_vector(env, info_new)
            
            # Store transition in replay buffer
            # Flatten terminated to a single boolean if it's an array
            if isinstance(terminated, (list, np.ndarray)):
                done = np.any(terminated)
            else:
                done = terminated
                
            if sum(rewards) >= -10:
                buffer.store_transition(obs_old, state_old, actions, rewards, obs_new, state_new, done)
            else:
                skipped_steps += 1
                print(f"Skipped step {skipped_steps} because of negative reward")
            
            # Update for next iteration
            obs = obs_new
            info = info_new
            
            t += 1
            total_steps += 1
            
            # Check if we've collected enough samples
            if buffer.mem_cntr >= buffer_size:
                break
        
        print(f"Episode completed after {t} steps")
        all_episode_rewards.append(episode_reward)
    
    print(f"Data collection completed after {total_steps} total steps")
    print(f"Buffer size: {buffer.mem_cntr}/{buffer_size}")
    print(f"Average reward: {sum(all_episode_rewards) / len(all_episode_rewards)}")
    # Save the buffer to a file
    print("Saving replay buffer to buffer.pkl...")
    with open('buffer.pkl', 'wb') as f:
        pickle.dump(buffer, f)
    print("Buffer saved successfully!")

def greedy_return_actions(obs, info, env):
    actions = []
    for agent_id in range(env.num_agents):
        # Get goal for the agent from centralized observation
        agent_goal, task = assign_goal(info, agent_id)
        
        # If agent has no goal, idle
        if agent_goal == -1:
            action = 6  # Idle action
        else:
            # Get agent's current position and orientation
            agent_pos = info['agents'][agent_id]['position']
            orientation = info['agents'][agent_id]['orientation']
            
            # Calculate the position in front of the agent
            dx, dy = DIRECTIONS[orientation]
            front_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
            
            # If the target is directly in front of the agent, perform pick/deliver action
            if front_pos == agent_goal:
                if task == 'pickup':
                    action = 4  # Pick
                elif task == 'delivery':
                    action = 5  # Deliver
                else:
                    action = 6  # Idle
            else:
                # Need to navigate to the goal
                action = greedy(info['grid'], agent_pos, agent_goal, agent_id, info)
        
        actions.append(action)

    return actions


if __name__ == "__main__":
    # Calculate metrics
    from metrics import calculate_metrics
    env = WarehouseEnv(
        num_agents=3, 
        grid_size=(15, 9), 
        goal_deliveries=10, 
        poisson_lambda=0.1, 
        max_requests=5,
        observation_type='centralized',
        normalize=True
    )
    num_episodes = 1000
    num_steps = 500
    throughput, utilization, task_completion_time, distances, avg_reward, collisions = calculate_metrics(env, greedy_return_actions, num_episodes, num_steps)
    print(f"Throughput: {throughput}")
    print(f"Utilization: {utilization}")
    print(f"Task completion time: {task_completion_time}")
    print(f"Distances: {distances}")
    print(f"Average reward: {avg_reward}")
    print(f"Collisions: {collisions}")