import numpy as np
import gymnasium as gym
import argparse
import copy
import multiprocessing
import signal
from gymnasium.spaces.utils import flatdim
from SimpleNEAT.NEAT import NEAT
from SimpleNEAT.utils import loadConfig

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


workerEnvironment = None
def initializeWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore C-c, parent will handle it

    global workerEnvironment
    workerEnvironment = CleanLunarLander(gym.make("LunarLanderContinuous-v3", render_mode=None))

def evaluateOrganism(organism, seeds):
    global workerEnvironment
    rewardsSum = 0
    for seed in seeds:
        state = workerEnvironment.reset(seed=int(seed))
        organism.clearMemory()
        
        while True:
            action = organism(state)
            state, reward, done = workerEnvironment.step(action)
            rewardsSum += reward
            if done: break
            
    return rewardsSum / len(seeds)

def main(config):
    rng = np.random.default_rng(config.seed)
    temporaryEnv = CleanLunarLander(gym.make("LunarLanderContinuous-v3", render_mode=None))
    solver = NEAT(config, inputSize=temporaryEnv.observationSize, outputSize=temporaryEnv.actionSize, rng=rng)
    temporaryEnv.close()
    
    population = solver.getInitialPopulation()

    generation = 0
    maxFitnessEver = -np.inf
    bestOrganism = None
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=initializeWorker) as pool:
        try:
            while True:
                seedsMatrix = rng.integers(0, 1000, size=(config.populationSize, config.evaluationEpisodes))

                result = pool.starmap_async(evaluateOrganism, zip(population, seedsMatrix))
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
                    break
                else:
                    population = solver.getNewPopulation(population, fitnessScores)
                    generation += 1

            # Visualize the winner
            env = CleanLunarLander(gym.make("LunarLanderContinuous-v3", render_mode="human"))
            for i in range(20):
                state = env.reset()
                bestOrganism.clearMemory()
                fitnessScore = 0
                while True:
                    action = bestOrganism(state)
                    state, reward, done = env.step(action)
                    fitnessScore += reward
                    if done: break
                print(f"Episode {i+1} Reward: {fitnessScore:.2f}")

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        finally:
            pool.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="lunarLander")
    main(loadConfig(parser.parse_args().config))