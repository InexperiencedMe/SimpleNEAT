import stable_retro as retro
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from SimpleNEAT.utils import loadConfig
from SimpleNEAT.trainer import runEvolution
from SimpleNEAT.showcaseOrganism import showcaseOrganism

class CleanMario(gym.Wrapper):
    def __init__(self, env):
        self.env = GrayscaleObservation(ResizeObservation(env, (16, 16)))
        self.observationSize = 16 * 16
        self.actionSize = 12

    def processObservation(self, obs):
        return obs.flatten() / 255.0

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self.processObservation(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step([1 if x > 0 else 0 for x in action])
        return self.processObservation(obs), reward, terminated or truncated
    
def environmentMaker(render_mode=None):
    return CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="DonutPlains1", render_mode=render_mode))

if __name__ == "__main__":
    config = loadConfig("superMarioWorld")
    bestOrganism = runEvolution(config, environmentMaker)
    showcaseOrganism(bestOrganism, environmentMaker, config.showcaseOptions)