import argparse
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import gymnasium as gym
from environmentUtils import CleanLunarLander
import copy
rng = np.random.default_rng(123)
novelSynapsesGlobal = {}  # {(source, destination): synapseID}
novelNeuronsGlobal  = {}  # {(source, destination): neuronID}
novelNeuronsCountGlobal = 10 # FIXME: Obviously it cannot stay like this, hardcoded LunarLander now

@dataclass(slots=True)
class Synapse:
    source:             int
    destination:        int
    weight:             float
    enabled:            bool

class Organism:
    def __init__(self, inputSize, outputSize):
        self.inputSize  = inputSize
        self.outputSize = outputSize
        self.neurons    = set(range(inputSize + outputSize))
        self.synapses   = {} # {synapseID: Synapse}
        self.memory     = self.memory = defaultdict(float)

        self.mutationChance_modifyWeight    = 0.8
        self.mutationChance_newSynapse      = 0.1
        self.mutationChance_newNeuron       = 0.03

    def clearMemory(self):
        self.memory.clear()

    def __call__(self, inputs): # NOTE: Mimic reference for now, then experiment with own ideas
        currentState = self.memory.copy() # TODO: Do I reeeaaallly need to copy?
        for i, input in enumerate(inputs):
            currentState[i] = input
        
        nextState = defaultdict(float)
        for synapse in self.synapses.values():
            if synapse.enabled:
                nextState[synapse.destination] += currentState[synapse.source] * synapse.weight

        for neuron in self.neurons:
            if neuron >= self.inputSize:
                self.memory[neuron] = np.tanh(nextState[neuron])
        
        return np.array([self.memory[self.inputSize + i] for i in range(self.outputSize)])

    def mutate(self):
        if rng.random() < self.mutationChance_modifyWeight:
            for synapse in self.synapses.values():
                if rng.random() < 0.9:
                    synapse.weight += rng.normal(0, 0.2)
                else:
                    synapse.weight = rng.normal(0, 1.0)

        if rng.random() < self.mutationChance_newSynapse:
            source        = rng.choice(list(self.neurons)) # TODO: We shouldnt include input to not skip this mutation
            destination   = rng.choice(list(self.neurons))

            if destination >= self.inputSize:
                key = (source, destination)
                existingConnections = set((synapse.source, synapse.destination) for synapse in self.synapses.values())

                if key not in existingConnections: # FIXME: Retry is better than this
                    if key not in novelSynapsesGlobal:
                        novelSynapsesGlobal[key] = len(novelSynapsesGlobal)

                    innovationID = novelSynapsesGlobal[key]
                    self.synapses[innovationID] = Synapse(source, destination, rng.normal(0, 1.0), True)
        
        if rng.random() < self.mutationChance_newNeuron and self.synapses:
            synapse = self.synapses[rng.choice(list(self.synapses.keys()))]

            if synapse.enabled: # FIXME: Retry if invalid synapse, otherwise skipped mutation
                synapse.enabled = False
                splitKey = (synapse.source, synapse.destination)

                if splitKey in novelNeuronsGlobal:
                    newNeuron = novelNeuronsGlobal[splitKey]
                else:
                    newNeuron = novelNeuronsCountGlobal
                    novelNeuronsCountGlobal += 1
                    novelNeuronsGlobal[splitKey] = newNeuron

                self.neurons.add(newNeuron)

                synapseNew1 = (synapse.source, newNeuron)
                if synapseNew1 not in novelSynapsesGlobal:
                    novelSynapsesGlobal[synapseNew1] = len(novelSynapsesGlobal)
                self.synapses[novelSynapsesGlobal[synapseNew1]] = Synapse(synapse.source, newNeuron, 1.0, True)

                synapseNew2 = (newNeuron, synapse.destination)
                if synapseNew2 not in novelSynapsesGlobal:
                    novelSynapsesGlobal[synapseNew2] = len(novelSynapsesGlobal)
                self.synapses[novelSynapsesGlobal[synapseNew2]] = Synapse(newNeuron, synapse.destination, synapse.weight, True)

    def reproduce(self, otherParent):
        child = Organism(self.inputSize, self.outputSize)
        child.neurons = set(self.neurons)

        for synapseID, synapse in self.synapses.items():
            if synapseID in otherParent.synapses:
                chosen = synapse if rng.random() > 0.5 else otherParent.synapses[synapseID]
                child.synapses[synapseID] = copy.deepcopy(chosen)
            else:
                child.synapses[synapseID] = copy.deepcopy(synapse) # Assume self is fitter, so we take their disjoint genes

        return child

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