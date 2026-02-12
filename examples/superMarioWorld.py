import stable_retro as retro
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from SimpleNEAT.utils import loadConfig
from SimpleNEAT.trainer import runEvolution
from SimpleNEAT.showcaseOrganism import showcaseOrganism

class CleanMario(gym.Wrapper):
    def __init__(self, env):
        self.env = GrayscaleObservation(env)
        self.observationSize = 16 * 14
        self.actionSize = 12
        
        self.max_Xposition      = 0
        self.stagnationCounter  = 0

    def getRAMvalues(self):
        ram = self.env.unwrapped.get_ram()

        Xposition = int(ram[0x94]) + (int(ram[0x95]) * 256) # low and high byte of X position
        isDead = (ram[0x71] == 9)                             # 9 = dying animation
        return Xposition, isDead

    def processObservation(self, obs):
        return obs[::16, ::16].flatten() / 255.0

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        
        Xposition, _ = self.getRAMvalues()
        self.max_Xposition = Xposition
        self.stagnationCounter = 0
        
        return self.processObservation(obs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step([1 if x > 0 else 0 for x in action])
        reward = 0

        current_X, isDead = self.getRAMvalues()
        progress = current_X - self.max_Xposition
        if progress > 0:
            self.max_Xposition = current_X
            self.stagnationCounter = 0
            reward += float(progress)
        else:
            self.stagnationCounter += 1
        reward += -1 # Constant penalty

        return self.processObservation(obs), reward, terminated or truncated or isDead or self.stagnationCounter >= 100
    
def environmentMaker(render_mode=None):
    return CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="DonutPlains1", render_mode=render_mode))

if __name__ == "__main__":
    config = loadConfig("superMarioWorld")
    bestOrganism = runEvolution(config, environmentMaker)
    showcaseOrganism(bestOrganism, environmentMaker, config.showcaseOptions)