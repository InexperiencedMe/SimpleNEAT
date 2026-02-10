import numpy as np
import multiprocessing
import signal
import copy
from SimpleNEAT.solver import NEAT

workerEnvironment = None
def initializeWorker(environmentMaker):
    global workerEnvironment
    workerEnvironment = environmentMaker()

    signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore C-c in workers. Main process handles that.

def evaluateOrganism(organism, seeds):
    global workerEnvironment
    rewardsSum = 0
    for seed in seeds:
        state = workerEnvironment.reset(seed=int(seed))
        organism.clearMemory()

        done = False
        while not done:
            action = organism(state)
            state, reward, done = workerEnvironment.step(action)
            rewardsSum += reward

    return rewardsSum / len(seeds)

def runEvolution(config, environmentMaker):
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
                seeds = rng.integers(0, 1000, size=(config.populationSize, config.evaluationEpisodes))
                fitnessScores = pool.starmap(evaluateOrganism, zip(population, seeds))
                
                indexBestThisGeneration = np.argmax(fitnessScores)
                scoreBestThisGeneration = fitnessScores[indexBestThisGeneration]
                if scoreBestThisGeneration > maxFitnessEver:
                    maxFitnessEver = scoreBestThisGeneration
                    bestOrganism = copy.deepcopy(population[indexBestThisGeneration])
                
                print(f"Generation {generation:>4}: Best this generation: {scoreBestThisGeneration:>8.2f} | Average: {np.mean(fitnessScores):8.2f} | Best Ever: {maxFitnessEver:8.2f}")

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
        
    return bestOrganism
