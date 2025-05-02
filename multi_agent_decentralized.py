from warehouse_env import WarehouseEnv
from maddpg import MADDPG
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def extract_full_state_vector(env, info):
    """
    Converts the full environment state into a single 1D vector,
    combining the flattened env.grid, shared requests (from one agent),
    and per-agent metadata (position, orientation, carrying).
    
    Args:
        env: The warehouse environment
        info: Dictionary containing state information from the environment
        
    Returns:
        np.ndarray: Flattened state vector for critic input
    """
    # 1. Flatten the global grid
    flattened_grid = env.grid.ravel().tolist()

    # 2. Extract shared request info (from first agent)
    shared_requests = []
    for req in info[0]['requests']:  # Use only once
        shared_requests.append(req['id'])                   # 1
        shared_requests.extend(req['source'].tolist())      # 2
        shared_requests.extend(req['destination'].tolist()) # 2
        shared_requests.append(req['picked'])               # 1

    # 3. Extract per-agent metadata (excluding requests)
    agent_metadata = []
    for agent_info in info:
        agent_metadata.extend(agent_info['position'].tolist())  # 2
        agent_metadata.append(agent_info['orientation'])        # 1
        agent_metadata.append(1 if np.any(agent_info['carrying']) else 0)  # 1

    # 4. Combine all parts
    full_vector = np.array(flattened_grid + agent_metadata + shared_requests, dtype=np.float32)

    return full_vector


def train_maddpg(env, agent, n_episodes=1000, max_steps=100, 
                 learn_interval=100, save_interval=50, eval_interval=20):
    """
    Train the MADDPG agent in the warehouse environment.
    
    Args:
        env: The warehouse environment
        agent: MADDPG agent
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learn_interval: Number of steps between learning updates
        save_interval: Number of episodes between model checkpoints
        eval_interval: Number of episodes between evaluations
        
    Returns:
        list: Episode rewards history
    """
    best_score = -np.inf
    score_history = []
    step_count = 0
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            state = extract_full_state_vector(env, info)
            actions = agent.choose_action(obs)
            
            # Convert actions to int if needed by environment
            actions = [int(a) for a in actions]
            
            obs_, reward, terminated, truncated, info = env.step(actions)
            state_ = extract_full_state_vector(env, info)
            
            # Store transition in memory
            agent.store_transition(obs, state, actions, reward, obs_, state_, 
                                  np.any(terminated))
            if np.any(terminated):
                print(f"Episode {episode} solved at step {step}")
            episode_reward += sum(reward)
            step_count += 1
            step += 1
            
            # Learn periodically
            if step_count % learn_interval == 0 and agent.memory.ready():
                agent.learn()
            
            obs = obs_
            
            if np.any(terminated) or np.any(truncated):
                break
                
        score_history.append(episode_reward)
        avg_score = np.mean(score_history[-100:])
        
        if episode % 100 == 0:
            print(f'Episode {episode}, Avg score: {avg_score:.2f}')
        
        # Save best model
        if avg_score > best_score and episode > 50:
            best_score = avg_score
            agent.save_checkpoint()
            
        # Regular checkpoints
        # if episode % save_interval == 0 and episode > 0:
        #     agent.save_checkpoint()
            
        # Evaluate periodically
        if episode % eval_interval == 0 and episode > 0:
            eval_score = evaluate_agent(env, agent, n_episodes=5)
            print(f'Evaluation at episode {episode}: {eval_score:.2f}')
    
    return score_history


def evaluate_agent(env, agent, n_episodes=10, max_steps=100, render=False):
    """
    Evaluate the trained agent without exploration.
    
    Args:
        env: The warehouse environment
        agent: MADDPG agent
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        
    Returns:
        float: Average total reward across evaluation episodes
    """
    total_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            if render:
                env.render()
                time.sleep(0.1)
                
            actions = agent.choose_action(obs, evaluate=True)  # No exploration
            actions = [int(a) for a in actions]
            
            obs_, reward, terminated, truncated, info = env.step(actions)
            
            episode_reward += sum(reward)
            step += 1
            obs = obs_
            
            if np.any(terminated) or np.any(truncated):
                break
                
        total_rewards.append(episode_reward)
        
    avg_reward = np.mean(total_rewards)
    return avg_reward


def plot_learning_curve(scores, filename):
    """
    Plot and save the learning curve.
    
    Args:
        scores: List of episode rewards
        filename: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = [i+1 for i in range(len(scores))]
    
    # Plot episode rewards
    ax.plot(x, scores, label='Episode Rewards')
    
    # Plot running average
    window_size = min(100, len(scores))
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-window_size):(i+1)])
    ax.plot(x, running_avg, label=f'Running Average ({window_size} episodes)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('MADDPG Learning Curve in Warehouse Environment')
    ax.legend()
    
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # Environment setup
    env_config = {
        'grid_size': (15, 9),
        'num_agents': 3,
        'normalize': True,
        'observation_type': 'decentralized'
    }
    
    env = WarehouseEnv(**env_config)
    obs, info = env.reset()
    
    # Calculate dimensions
    actor_dims = [env.observation_space.shape[1]] * env.num_agents
    critic_dims = extract_full_state_vector(env, info).shape[0]
    num_agents = env.num_agents
    n_actions = env.action_space[0].n
    
    print(f"Actor dimensions: {actor_dims}")
    print(f"Critic dimensions: {critic_dims}")
    print(f"Number of agents: {num_agents}")
    print(f"Action space size: {n_actions}")
    
    # MADDPG agent setup
    maddpg_config = {
        'actor_dims': actor_dims,
        'critic_dims': critic_dims,
        'n_actions': n_actions,
        'n_agents': num_agents,
        'alpha': 0.0005,      # Actor learning rate
        'beta': 0.0005,       # Critic learning rate
        'fc1': 256,         # First hidden layer size
        'fc2': 128,          # Second hidden layer size
        'gamma': 0.99,      # Discount factor
        'tau': 0.01,        # Target network update rate
        'batch_size': 128,
        'chkpt_dir': 'checkpoints6/',
        'buffer_path': '/Users/mayankchandak/Desktop/MY STUFF/Courses/SEM6/Automan/Automan_project/ME3302_Warehouse_Management/buffer.pkl',
    }
    
    agent = MADDPG(**maddpg_config)
    
    # Training
    print("Starting training...")
    training_config = {
        'n_episodes': 1000,
        'max_steps': 1500,
        'learn_interval': 1,
        'save_interval': 100,
        'eval_interval': 100
    }
    
    score_history = train_maddpg(env, agent, **training_config)
    
    # Plot learning curve
    plot_learning_curve(score_history, 'warehouse_maddpg_learning_curve.png')
    
    # Final evaluation
    print("Final evaluation...")
    eval_score = evaluate_agent(env, agent, n_episodes=10, render=True)
    print(f"Final evaluation score: {eval_score:.2f}")
    
    env.close()