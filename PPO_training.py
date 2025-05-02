from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from warehouse_env import WarehouseEnv

# Define a callable that returns a new environment instance
env_fn = lambda: WarehouseEnv(observation_type='centralized', normalize=True, grid_size=(15, 9), num_agents=3)

# # Wrap in DummyVecEnv
# vec_env = DummyVecEnv([env_fn])

# # Optional: check single env, not vectorized one
# check_env(env_fn())

# # Now train using vec_env
# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=20_000_000)

# # Save
# model.save("ppo_warehouse_2")

# Reload env and model
vec_env = DummyVecEnv([env_fn])
model = PPO.load("ppo_warehouse_2")
print(vec_env.observation_space, vec_env.action_space)
# Reset
obs = vec_env.reset()
for _ in range(100):
    action, _ = model.predict(obs)  # action shape: (n_envs, num_agents)
    obs, reward, done, info = vec_env.step(action)

    # Optional: render raw env
    vec_env.envs[0].render()  # access the underlying env
    if done:
        obs = vec_env.reset()
