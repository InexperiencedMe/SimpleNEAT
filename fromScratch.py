from dataclasses import dataclass
from collections import defaultdict
import numpy as np
rng = np.random.default_rng(123)

INPUT_SIZE, OUTPUT_SIZE = 8, 2

novelSynapsesGlobal = {}
neuronCountGlobal   = INPUT_SIZE + OUTPUT_SIZE

@dataclass
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
        self.fitness         = 0.0
        self.fitnessAdjusted = 0.0
        self.memory = defaultdict(float)
        self.mutationChance_modWeight   = 0.8
        self.mutationChance_newSynapse  = 0.1
        self.mutationChance_newNeuron   = 0.03

    def clearMemory(self):
        self.memory.clear()

    def __call__(self, inputs):
        for i, input in enumerate(inputs):
            self.memory[i] = input

        nextState = defaultdict(float)
        for synapse in self.synapses:
            if synapse.enabled:
                nextState[synapse.destinationNeuron] += np.tanh(self.memory[synapse.sourceNeuron] * synapse.weight)

        self.memory = nextState
        return [nextState[self.inputSize + i] for i in range(self.outputSize)]
    
    def mutate(self):
        mutationRoll = rng.random()

        if mutationRoll < self.mutationChance_modWeight:
            for synapse in self.synapses.values():
                if rng.random() < 0.9:
                    synapse.weight += rng.normal(0, 0.2)
                else:
                    synapse.weight = rng.normal(0, 1.0)
        
        if mutationRoll < self.mutationChance_newSynapse:
            sourceNeuronNew, destinationNeuronNew = rng.choice(list(self.neurons)), rng.choice(list(self.neurons))
            if destinationNeuronNew >= self.inputSize: # no connection to outputs
                key = (sourceNeuronNew, destinationNeuronNew)
                existing = set((synapse.sourceNeuron, synapse.destinationNeuron) for synapse in self.synapses.values())
                if key not in existing:
                    if key not in novelSynapsesGlobal:
                        novelSynapsesGlobal[key] = len(novelSynapsesGlobal)
                    self.synapses[novelSynapsesGlobal[key]] = Synapse(sourceNeuronNew, destinationNeuronNew, rng.normal(0, 1.0), True)
        
        if mutationRoll < self.mutationChance_newNeuron and self.synapses:
            synapse = self.synapses[rng.choice(list(self.synapses.keys()))]
            if synapse.enabled:
                synapse.enabled = False
                newNeuron = neuronCountGlobal
                neuronCountGlobal += 1
                self.neurons.add(newNeuron)

                synapseNew1 = (synapse.sourceNeuron, newNeuron)
                if synapseNew1 not in novelSynapsesGlobal:
                    novelSynapsesGlobal[synapseNew1] = len(novelSynapsesGlobal)
                self.synapses[novelSynapsesGlobal[synapseNew1]] = Synapse(synapse.sourceNeuron, newNeuron, 1.0, True)

                synapseNew2 = (newNeuron, synapse.destinationNeuron)
                if synapseNew2 not in novelSynapsesGlobal:
                    novelSynapsesGlobal[synapseNew2] = len(novelSynapsesGlobal)
                self.synapses[novelSynapsesGlobal[synapseNew2]] = Synapse(newNeuron, synapse.destinationNeuron, synapse.weight, True)

    def distance(self, other):
        synapseIDsSelf, synapseIDsOther = set(self.synapses.keys()), set(other.synapses.keys())
        disjointNumber = len(synapseIDsSelf ^ synapseIDsOther)
        matching = synapseIDsSelf & synapseIDsOther
        weightsDifference = sum(abs(self.synapses[synapseID].weight - other.synapses[synapseID].weight) for synapseID in matching)/len(matching) if matching else 0
        return 1*disjointNumber + 0.4*weightsDifference