import argparse
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import gymnasium as gym
from environmentUtils import CleanLunarLander
rng = np.random.default_rng(123)

@dataclass(slots=True)
class Synapse:
    sourceNeuron:       int
    destinationNeuron:  int
    weight:             float
    enabled:            bool

class Organism:
    def __init__(self, inputSize, outputSize):
        self.inputSize  = inputSize
        self.outputSize = outputSize
        self.neurons    = set(range(inputSize + outputSize))
        self.synapses   = {} # {innovationID: Synapse}
        self.memory     = self.memory = defaultdict(float)

        self.mutationChance_modifyWeight    = 0.8
        self.mutationChance_newSynapse      = 0.1
        self.mutationChance_newNeuron       = 0.03

    def clearMemory(self):
        self.memory.clear()

    def __call__(self, input):
        pass

    def mutate(self):
        if rng.random() < self.mutationChance_modifyWeight:
            for synapse in self.synapses.values():
                if rng.random() < 0.9:
                    synapse.weight += rng.normal(0, 0.2)
                else:
                    synapse.weight = rng.normal(0, 1.0)

        if rng.random() < self.mutationChance_newSynapse:
            pass

        if rng.random() < self.mutationChance_newNeuron and self.synapses:
            pass

    def reproduce(otherParent):
        pass

class NEAT:
    def __init__(self, populationSize, inputSize, outputSize):
        self.populationSize = populationSize
        self.inputSize      = inputSize
        self.outputSize     = outputSize

    def getInitialPopulation(self):
        pass

    def getNewPopulation(self, population, fitnessScores):
        pass

    def calculateGeneticDistance(self, firstOrganism, secondOrganism):
        pass

def main(args):
    env = CleanLunarLander(gym.make("LunarLanderContinuous-v3", render_mode=None))
    solver = NEAT(args.populationSize, inputSize=env.observationSize, outputSize=env.actionSize)
    population = solver.getInitialPopulation()

    fitnessScores = np.zeros(args.populationSize, dtype=np.float32)
    while True:
        for i, organism in enumerate(population):
            fitnessScore = 0
            for _ in range(args.evaluationEpisodes):
                state = env.reset()
                organism.clearMemory()
                while True:
                    action = organism(state)
                    state, reward, done = env.step(action)
                    fitnessScore += reward
                    if done: break
            fitnessScores[i] = fitnessScore / args.evaluationEpisodes
        
        endConditionMet = np.max(fitnessScores) >= args.targetFitness
        if endConditionMet:
            break
        else:
            population = solver.getNewPopulation(population, fitnessScores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",  "--runName",                 type=str,   default="myNEATrun")
    parser.add_argument("-p",  "--populationSize",          type=int,   default=150)
    parser.add_argument("-t",  "--targetFitness",           type=float, default=300.0)
    parser.add_argument("-ee", "--evaluationEpisodes",      type=int,   default=3)

    main(parser.parse_args())