import gymnasium as gym
import numpy as np
from SimpleNEAT.utils import loadConfig
from SimpleNEAT.trainer import runEvolution
from SimpleNEAT.showcaseOrganism import showcaseOrganism

class CleanLunarLander(gym.Wrapper):
    def __init__(self, render_mode=None):
        self.env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)
        self.actionSize         = gym.spaces.utils.flatdim(self.env.action_space)
        self.observationShape   = (gym.spaces.utils.flatdim(self.env.observation_space),)
        self.observationSize    = np.prod(self.observationShape)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done

    def reset(self, seed=None):
        observation, info = self.env.reset(seed=seed)
        return observation

def environmentMaker(render_mode=None):
    return CleanLunarLander(render_mode=render_mode)

if __name__ == "__main__":
    config = loadConfig("lunarLander")
    bestOrganism, solver = runEvolution(config, environmentMaker)
    showcaseOrganism(bestOrganism, solver, environmentMaker, config.showcaseOptions)