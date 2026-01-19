from dataclasses import dataclass
from collections import defaultdict
import numpy as np

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