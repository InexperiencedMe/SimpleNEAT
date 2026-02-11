import stable_retro as retro
import gymnasium as gym
from SimpleNEAT.utils import loadConfig
from SimpleNEAT.trainer import runEvolution
from SimpleNEAT.showcaseOrganism import showcaseOrganism

class CleanMario(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.observationSize = 16 * 14 
        self.actionSize = 12 
        
        self.time_since_last_reward = 0
        self.current_total_reward = 0.0

    def process_obs(self, obs):
        downsampled = obs[::16, ::16, :3]
        gray = (downsampled[:,:,0] * 0.2989 + downsampled[:,:,1] * 0.5870 + downsampled[:,:,2] * 0.1140)
        return gray.flatten() / 255.0

    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)
            
        self.time_since_last_reward = 0
        self.current_total_reward = 0.0
        return self.process_obs(state)

    def step(self, action):
        binary_action = [1 if x > 0 else 0 for x in action]
        obs, reward, terminated, truncated, info = self.env.step(binary_action)
        
        self.current_total_reward += reward

        if reward > 0:
            self.time_since_last_reward = 0
        else:
            self.time_since_last_reward += 1
            
        is_stagnant = self.time_since_last_reward > 1000
        
        done = terminated or truncated or is_stagnant
        return self.process_obs(obs), reward, done

def environmentMaker(render_mode=None):
    return CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="DonutPlains1", render_mode=render_mode))

if __name__ == "__main__":
    config = loadConfig("superMarioWorld")
    bestOrganism = runEvolution(config, environmentMaker)
    showcaseOrganism(bestOrganism, environmentMaker, config.showcaseOptions)