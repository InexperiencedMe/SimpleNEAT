import numpy as np
import multiprocessing
import signal
import copy
import pickle
import os
from SimpleNEAT.NEAT import NEAT
from SimpleNEAT.utils import ensurePath

workerEnvironment = None
def initializeWorker(environmentMaker):
    global workerEnvironment
    workerEnvironment = environmentMaker()

    signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore C-c in workers. Main process handles that.

def evaluateOrganism(organism, seeds):
    global workerEnvironment
    rewardsSum = 0
    for seed in seeds:
        observation = workerEnvironment.reset(seed=int(seed))
        organism.clearMemory()

        done = False
        while not done:
            action = organism(observation)
            observation, reward, done = workerEnvironment.step(action)
            rewardsSum += reward

    return rewardsSum / len(seeds)

def saveCheckpoint(runName, population, solver, bestOrganism, generation, maxFitnessEver):
    filepath = ensurePath("checkpoints", runName, f"Gen_{generation}-Fitness_{maxFitnessEver:.0f}.pkl")
    state = {
        'generation'        : generation,
        'maxFitnessEver'    : maxFitnessEver,
        'bestOrganism'      : bestOrganism,
        'population'        : population,
        'solver'            : solver}
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    print(f"Checkpoint saved to: {filepath}")

def loadCheckpoint(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def runEvolution(config, environmentMaker, resumePath=None):
    if resumePath is not None and os.path.exists(resumePath):
        print(f"Resuming training from checkpoint: {resumePath}")
        state = loadCheckpoint(resumePath)
        generation     = state['generation']
        maxFitnessEver = state['maxFitnessEver']
        bestOrganism   = state['bestOrganism']
        population     = state['population']
        solver         = state['solver']
        
        solver.config = config
        for org in population:
            org.config = config
            
    else:
        rng = np.random.default_rng(config.seed)
        
        temporaryEnvironment = environmentMaker()
        solver = NEAT(config, inputSize=temporaryEnvironment.observationSize, outputSize=temporaryEnvironment.actionSize, rng=rng)
        temporaryEnvironment.close()

        population = solver.getInitialPopulation()

        generation      = 0
        maxFitnessEver  = -np.inf
        bestOrganism    = None

    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=initializeWorker, initargs=(environmentMaker,)) as pool:
        try:
            while True:
                seeds = solver.rng.integers(0, 1000, size=(config.populationSize, config.evaluationEpisodes))
                fitnessScores = pool.starmap(evaluateOrganism, zip(population, seeds))
                
                indexBestThisGeneration = np.argmax(fitnessScores)
                scoreBestThisGeneration = fitnessScores[indexBestThisGeneration]

                if scoreBestThisGeneration > maxFitnessEver:
                    maxFitnessEver = scoreBestThisGeneration
                    bestOrganism = copy.deepcopy(population[indexBestThisGeneration])

                    if config.saveCheckpoints:
                        saveCheckpoint(config.runName, population, solver, bestOrganism, generation, maxFitnessEver)
                
                print(f"Generation {generation:>4}: "
                    f"Best this generation: {scoreBestThisGeneration:>8.2f} | "
                    f"Average: {np.mean(fitnessScores):8.2f} | "
                    f"Best Ever: {maxFitnessEver:8.2f}")


                if maxFitnessEver >= config.targetFitness:
                    print("Target fitness reached!")
                    break
                else:
                    population = solver.getNewPopulation(population, fitnessScores)
                    generation += 1
                
        except KeyboardInterrupt:
            print("\nTraining terminated early by the user")
            pool.terminate()
            pool.join()
        finally:
            pool.close()
        
    return bestOrganism, solver
