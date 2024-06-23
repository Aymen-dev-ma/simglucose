import gym
import numpy as np
from stable_baselines3 import PPO
from simglucose.envs import T1DSimEnv

import matplotlib.pyplot as plt

# Define your custom environment
class MyEnvironment(T1DSimEnv):
    def __init__(self):
        super().__init__()

    def reset(self):
        return super().reset()

    def step(self, action):
        return super().step(action)

    def render(self):
        super().render()

# Create an instance of your custom environment
env = MyEnvironment()

# Train the environment using stable-baselines3 PPO algorithm
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Visualize the training progress
plt.plot(model.ep_info_buffer)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Training Progress")
plt.show()