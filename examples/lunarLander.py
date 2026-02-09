import numpy as np
import gymnasium as gym
import argparse
import copy
import multiprocessing
import signal
from gymnasium.spaces.utils import flatdim
from SimpleNEAT.NEAT import NEAT

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

def main(args):
    rng = np.random.default_rng(args.seed)
    temporaryEnv = CleanLunarLander(gym.make("LunarLanderContinuous-v3", render_mode=None))
    solver = NEAT(args, inputSize=temporaryEnv.observationSize, outputSize=temporaryEnv.actionSize, rng=rng)
    temporaryEnv.close()
    
    population = solver.getInitialPopulation()

    generation = 0
    maxFitnessEver = -np.inf
    bestOrganism = None
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=initializeWorker) as pool:
        try:
            while True:
                seedsMatrix = rng.integers(0, 1000, size=(args.populationSize, args.evaluationEpisodes))

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

                if maxFitnessEver >= args.targetFitness:
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
    parser.add_argument("-n",  "--runName",                       type=str,     default="lunarLanderNEAT")
    parser.add_argument("-s",  "--seed",                          type=int,     default=123)
    parser.add_argument("-t",  "--targetFitness",                 type=float,   default=320.0)
    parser.add_argument("-p",  "--populationSize",                type=int,     default=150)
    parser.add_argument("-ss", "--targetSpeciesSize",             type=int,     default=15)
    parser.add_argument("-st", "--survivalThreshold",             type=float,   default=0.2)
    parser.add_argument("-e",  "--elitism",                       type=int,     default=2)
    parser.add_argument("-sg", "--stagnationThreshold",           type=int,     default=50)
    parser.add_argument("-ee", "--evaluationEpisodes",            type=int,     default=3)
    parser.add_argument("-ct", "--defaultCompatibilityThreshold", type=float,   default=3.0)
    parser.add_argument("-cs", "--compatibilityAdjustmentSpeed",  type=float,   default=0.2)
    parser.add_argument("-le", "--lossWeightExcess",              type=float,   default=1.0)
    parser.add_argument("-ld", "--lossWeightDisjoint",            type=float,   default=1.0)
    parser.add_argument("-lw", "--lossWeightWeightsDifference",   type=float,   default=0.0)
    parser.add_argument("-mw", "--mutationChanceModifyWeight",    type=float,   default=0.5)
    parser.add_argument("-ms", "--mutationChanceNewSynapse",      type=float,   default=0.1)
    parser.add_argument("-mn", "--mutationChanceNewNeuron",       type=float,   default=0.05)
    parser.add_argument("-mr", "--resetWeightChance",             type=float,   default=0.1)
    parser.add_argument("-ws", "--weightMutationScale",           type=float,   default=0.01)
    main(parser.parse_args())