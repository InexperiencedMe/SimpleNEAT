import gymnasium as gym
from gymnasium.spaces.utils import flatdim

class CleanLunarLander(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observationSize = flatdim(self.env.observation_space)
        self.actionSize      = flatdim(self.env.action_space)


    def step(self, action):
        observations, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observations, reward, done

    def reset(self, seed=None):
        observations, info = self.env.reset(seed=seed)
        return observations