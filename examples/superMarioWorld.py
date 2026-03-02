import stable_retro as retro
import gymnasium as gym
import numpy as np
import cv2
from SimpleNEAT.utils import loadConfig
from SimpleNEAT.trainer import runEvolution
from SimpleNEAT.showcaseOrganism import showcaseOrganism

class CleanMario(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.actionSize = 6 
        self.observationShape = (8, 8)
        self.observationSize = np.prod(self.observationShape)
        
        originalShape = env.observation_space.shape
        self.cropTop = int(originalShape[0] * 0.5)
        self.cropLeft = int(originalShape[1] * 0.5)
        
        self.maxXposition = 0
        self.stagnationCounter = 0

    def getRAMvalues(self):
        ram = self.env.unwrapped.get_ram()
        xPosition = int(ram[0x94]) + (int(ram[0x95]) * 256) 
        isDead = (ram[0x71] == 9)                             
        return xPosition, isDead

    def processObservation(self, obs):
        croppedObs = obs[self.cropTop:, self.cropLeft:]
        greenObs = croppedObs[:, :, 1] 
        targetSize = (self.observationShape[1], self.observationShape[0])
        resizedObs = cv2.resize(greenObs, targetSize, interpolation=cv2.INTER_AREA)
        
        return (resizedObs / 127.5) - 1.0

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        xPosition, _ = self.getRAMvalues()
        self.maxXposition = xPosition
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

        currentX, isDead = self.getRAMvalues()
        progress = currentX - self.maxXposition
        
        if progress > 0:
            self.maxXposition = currentX
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
    bestOrganism, solver = runEvolution(config, environmentMaker, resumePath="checkpoints/superMarioWorldNEAT-8x8-SUPERFIXED3/Gen_3493-Fitness_2778.pkl")
    showcaseOrganism(bestOrganism, solver, environmentMaker, config.showcaseOptions)