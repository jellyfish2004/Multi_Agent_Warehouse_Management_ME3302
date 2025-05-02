from warehouse_env import WarehouseEnv
from maddpg import MADDPG
import numpy as np
import time
def evaluate_agent(env, agent, n_episodes=10, max_steps=1500, render=False):
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
                #time.sleep(0.5)
                
            actions = agent.choose_action(obs, evaluate=True)  # No exploration
            actions = [int(a) for a in actions]
            
            obs_, reward, terminated, truncated, info = env.step(actions)
            print(actions)
            episode_reward += sum(reward)
            step += 1
            obs = obs_
            
            if np.any(terminated) or np.any(truncated):
                break
                
        total_rewards.append(episode_reward)
        
    avg_reward = np.mean(total_rewards)
    return avg_reward



if __name__ == "__main__":
    # Environment setup
    env_config = {
        'grid_size': (15, 9),
        'num_agents': 3,
        'normalize': True,
        #'observation_type': 'decentralized_partial_observability'
        'observation_type': 'decentralized',
    }
    
    env = WarehouseEnv(**env_config)
    obs, info = env.reset()
    
    # Calculate dimensions
    actor_dims = [env.observation_space.shape[1]] * env.num_agents
    critic_dims = 177
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
        'alpha': 0.005,      # Actor learning rate
        'beta': 0.005,       # Critic learning rate
        'fc1': 256,         # First hidden layer size
        'fc2': 128,          # Second hidden layer size
        'gamma': 0.99,      # Discount factor
        'tau': 0.01,        # Target network update rate
        'batch_size': 128,
        'chkpt_dir': 'checkpoints5/'
    }
    
    agent = MADDPG(**maddpg_config)
    agent.load_checkpoint()
    
    eval_score = evaluate_agent(env, agent, n_episodes=5, render=True)
    print(f"evaluation score: {eval_score:.2f}")
    
    env.close()