import numpy as np
import stable_retro as retro
import argparse
import copy
import multiprocessing
import signal
import pygame as pg
import gymnasium as gym
import itertools
from SimpleNEAT.NEAT import NEAT
from SimpleNEAT.utils import loadConfig

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
            
        is_stagnant = self.time_since_last_reward > 300
        
        done = terminated or truncated or is_stagnant
        return self.process_obs(obs), reward, done

workerEnvironment = None
def initializeWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore C-c, parent will handle it

    global workerEnvironment
    workerEnvironment = CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="DonutPlains1", render_mode=None))

def evaluateOrganism(organism, numberOfEpisodes):
    global workerEnvironment
    rewardsSum = 0
    for _ in range(numberOfEpisodes):
        state = workerEnvironment.reset()
        organism.clearMemory()
        
        while True:
            action = organism(state)
            state, reward, done = workerEnvironment.step(action)
            rewardsSum += reward
            if done: break
    return rewardsSum / numberOfEpisodes

def main(config):
    rng = np.random.default_rng(config.seed)
    temporaryEnv = CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="DonutPlains1", render_mode=None))
    solver = NEAT(config, inputSize=temporaryEnv.observationSize, outputSize=temporaryEnv.actionSize, rng=rng)
    temporaryEnv.close()
    
    population = solver.getInitialPopulation()

    generation      = 0
    maxFitnessEver  = -np.inf
    bestOrganism    = None
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=initializeWorker) as pool:
        try:
            while True:
                result = pool.starmap_async(evaluateOrganism, zip(population, itertools.repeat(config.evaluationEpisodes)))
                fitnessScores = result.get(timeout=999999) 
                
                maxFitnessThisGeneration = -np.inf
                for i, evaluatedFitness in enumerate(fitnessScores):
                    if evaluatedFitness > maxFitnessEver:
                        maxFitnessEver = evaluatedFitness
                        bestOrganism = copy.deepcopy(population[i])
                    if evaluatedFitness > maxFitnessThisGeneration:
                        maxFitnessThisGeneration = evaluatedFitness

                avgFitness = np.mean(fitnessScores)
                print(f"Generation {generation:4}: Best this generation: {maxFitnessThisGeneration:>8.2f} | Average: {avgFitness:8.2f} | Best Ever: {maxFitnessEver:8.2f}")

                if maxFitnessEver >= config.targetFitness:
                    print("Target fitness reached!")
                    break
                else:
                    population = solver.getNewPopulation(population, fitnessScores)
                    generation += 1

        except KeyboardInterrupt:
            print("Training terminated early by the user")
            pool.terminate()
            pool.join()
        finally:
            pool.close()

    # Visualize the winner
    pg.init()
    win = pg.display.set_mode((768, 672))
    pg.display.set_caption("NEAT Mario Replay")
    clock = pg.time.Clock()
    
    env = CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="DonutPlains1", render_mode="rgb_array"))
    quitReplay = False
    while not quitReplay:
        state = env.reset()
        bestOrganism.clearMemory()
        fitnessScore = 0
        while True:
            frame = env.render()
            surface = pg.surfarray.make_surface(frame.swapaxes(0, 1))
            win.blit(pg.transform.scale(surface, (768, 672)), (0, 0))
            pg.display.flip()

            action = bestOrganism(state)
            state, reward, done = env.step(action)
            
            fitnessScore += reward
            
            for event in pg.event.get():
                if event.type == pg.QUIT: quitReplay = True
            
            if done or quitReplay: 
                break

            clock.tick(60)
        print(f"Best organism showcase reward: {fitnessScore:.2f}")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="superMarioWorld")
    main(loadConfig(parser.parse_args().config))