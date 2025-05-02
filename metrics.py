import numpy as np

def calculate_metrics(env, func, num_episodes=100, num_steps=500):
    # Calculate throughput, utilization, task completion time, total distance, avg reward, collisions
    throughput = []
    utilization = []
    task_completion_time = []
    distances = []
    avg_reward = []
    collisions = []
    
    
    for i in range(num_episodes):
        obs, info_ = env.reset()
        info = env._obs_centralized()
        n_steps = 0
        time_with_load = [0] * env.num_agents
        episode_reward = 0
        request_completion_time = {}
        episode_distance = 0
        for j in range(num_steps):
            actions = func(obs, info_, env)
            obs, rewards, terminated, truncated, info_ = env.step(actions)
            info = env._obs_centralized()
            n_steps += 1
            for i, agent in enumerate(info['agents']):
                if agent['carrying'] != -1:
                    time_with_load[i] += 1
            episode_reward += rewards / env.num_agents
            for req in info['requests']:
                if req['id'] != -1 and req['id'] not in request_completion_time:
                    request_completion_time[req['id']] = 0
                elif req['id'] in request_completion_time:
                    request_completion_time[req['id']] += 1
            for act in actions:
                if act in [0, 1]:
                    episode_distance += 1
            if terminated or truncated:
                break
        throughput.append(env.deliveries_completed / n_steps)
        utilization.append(sum([t_load / n_steps for t_load in time_with_load]) / env.num_agents)
        task_completion_time.append(sum([request_completion_time[req_id] for req_id in request_completion_time]) / len(request_completion_time))
        distances.append(episode_distance)
        avg_reward.append(episode_reward)
        collisions.append(env.num_collisions)
    return sum(throughput) / num_episodes, sum(utilization) / num_episodes, sum(task_completion_time) / num_episodes, sum(distances) / num_episodes, sum(avg_reward) / num_episodes, sum(collisions) / num_episodes


        
        