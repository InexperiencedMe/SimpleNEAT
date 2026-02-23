import stable_retro as retro
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from SimpleNEAT.utils import loadConfig
from SimpleNEAT.trainer import runEvolution
from SimpleNEAT.showcaseOrganism import showcaseOrganism

class CleanMario(gym.Wrapper):
    def __init__(self, env):
        self.actionSize = 6 
        self.observationShape = (14, 16)
        self.observationSize = np.prod(self.observationShape)

        self.env = ResizeObservation(GrayscaleObservation(env), self.observationShape)
        
        self.max_Xposition      = 0
        self.stagnationCounter  = 0

    def getRAMvalues(self):
        ram = self.env.unwrapped.get_ram()
        Xposition = int(ram[0x94]) + (int(ram[0x95]) * 256) 
        isDead = (ram[0x71] == 9)                             
        return Xposition, isDead

    def processObservation(self, obs):
        return (obs / 127.5) - 1

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        Xposition, _ = self.getRAMvalues()
        self.max_Xposition = Xposition
        self.stagnationCounter = 0
        return self.processObservation(obs)

    def step(self, action):
        fullAction = [0] * 12
        fullAction[0] = 1 if action[0] > 0 else 0 # B (Jump)
        fullAction[1] = 1 if action[1] > 0 else 0 # Y (Run/Shoot)
        fullAction[5] = 1 if action[2] > 0 else 0 # Down
        fullAction[6] = 1 if action[3] > 0 else 0 # Left
        fullAction[7] = 1 if action[4] > 0 else 0 # Right
        fullAction[8] = 1 if action[5] > 0 else 0 # A (Spin Jump)

        obs, _, terminated, truncated, info = self.env.step(fullAction)
        reward = 0

        current_X, isDead = self.getRAMvalues()
        progress = current_X - self.max_Xposition
        if progress > 0:
            self.max_Xposition = current_X
            self.stagnationCounter = 0
            reward += float(progress)
        else:
            self.stagnationCounter += 1
        reward -= 1 # Constant penalty

        return self.processObservation(obs), reward, terminated or truncated or isDead or self.stagnationCounter >= 100
    
def environmentMaker(render_mode=None):
    return CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="DonutPlains1", render_mode=render_mode))

if __name__ == "__main__":
    config = loadConfig("superMarioWorld")
    bestOrganism, solver = runEvolution(config, environmentMaker)
    showcaseOrganism(bestOrganism, solver, environmentMaker, config.showcaseOptions)