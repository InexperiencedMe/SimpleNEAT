import numpy as np
import copy
from dataclasses import dataclass
from collections import defaultdict

@dataclass(slots=True)
class Synapse:
    source:      int
    destination: int
    weight:      float
    enabled:     bool

    def __str__(self) -> str:
        return f"""
        Synapse(source      = {self.source}
                weight      = {self.weight:.2f}
                destination = {self.destination}
                enabled     = {self.enabled})"""

class Organism:
    def __init__(self, config, inputSize, outputSize, rng):
        self.config             = config
        self.inputSize          = inputSize
        self.inputSizeWithBias  = inputSize + 1
        self.biasNeuron         = inputSize
        self.outputSize         = outputSize
        self.neurons     = set(range(self.inputSizeWithBias + self.outputSize))
        self.synapses    = {} # {synapseID: Synapse}
        self.memory      = defaultdict(float)
        self.lastSignals = defaultdict(float) # Not a fan, but I need exact signals for visualization
        self.rng         = rng

    def clearMemory(self):
        self.memory.clear()
        self.lastSignals.clear()

    def __call__(self, inputs):
        if inputs.ndim > 1: inputs = inputs.ravel()
        
        for i, input in enumerate(inputs):
            self.memory[i] = input
        self.memory[self.biasNeuron] = 1.0
        
        newState = defaultdict(float)
        for synapseID, synapse in self.synapses.items():
            if synapse.enabled:
                signal = self.memory[synapse.source] * synapse.weight
                newState[synapse.destination] += signal
                self.lastSignals[synapseID] = signal

        for neuronID in self.neurons:
            if neuronID >= self.inputSizeWithBias: 
                self.memory[neuronID] = np.tanh(newState[neuronID])
        
        return np.array([self.memory[self.inputSizeWithBias + i] for i in range(self.outputSize)])
    
    def initializeSynapses(self, tracker):
        for input in range(self.inputSizeWithBias):
            for output in range(self.inputSizeWithBias, self.inputSizeWithBias + self.outputSize):
                self.synapses[tracker.getSynapseID(input, output)] = Synapse(input, output, self.rng.normal(0, 1.0), True)

    def mutate(self, tracker):
        if self.rng.random() < self.config.mutationChanceModifyWeight:
            for synapse in self.synapses.values():
                if self.rng.random() < self.config.resetWeightChance:
                    synapse.weight = self.rng.normal(0, 1.0)
                else:
                    synapse.weight += self.rng.normal(0, self.config.weightMutationScale)

        if self.rng.random() < self.config.mutationChanceSynapse:
                validSources        = list(self.neurons)
                validDestinations   = [n for n in self.neurons if n >= self.inputSizeWithBias]
                existingLinks       = set((synapse.source, synapse.destination) for synapse in self.synapses.values())
                for _ in range(10): # Retrying is faster than listing possible new links
                    source        = int(self.rng.choice(validSources))
                    destination   = int(self.rng.choice(validDestinations))
                    link = (source, destination)
                    if link not in existingLinks:
                        self.synapses[tracker.getSynapseID(source, destination)] = Synapse(source, destination, self.rng.normal(0, 1.0), True)
                        break
        
        if self.rng.random() < self.config.mutationChanceNeuron:
            if self.rng.random() < 0.5: # Add a neuron
                if self.synapses:
                    synapseKeys = list(self.synapses.keys())
                    for _ in range(10):
                        synapseToSplit = self.synapses[self.rng.choice(synapseKeys)]
                        if synapseToSplit.enabled:
                            # TODO: If the parallel synapse stays, delete synapse.enabled property globally
                            # synapseToSplit.enabled = False

                            newNeuron = tracker.getNeuronID(synapseToSplit.source, synapseToSplit.destination)
                            if newNeuron in self.neurons: continue
                            self.neurons.add(newNeuron)

                            newLinkID1 = tracker.getSynapseID(synapseToSplit.source, newNeuron)
                            self.synapses[newLinkID1] = Synapse(synapseToSplit.source, newNeuron, 1.0, True)

                            newLinkID2 = tracker.getSynapseID(newNeuron, synapseToSplit.destination)
                            self.synapses[newLinkID2] = Synapse(newNeuron, synapseToSplit.destination, 0.0, True)
                            break
            else: # Remove a neuron
                hiddenNeurons = [n for n in self.neurons if n >= self.inputSizeWithBias + self.outputSize]
                if hiddenNeurons:
                    neuronToRemove = int(self.rng.choice(hiddenNeurons))
                    self.neurons.remove(neuronToRemove)
                    
                    synapsesToRemove = [id for id, s in self.synapses.items() if s.source == neuronToRemove or s.destination == neuronToRemove]
                    for synapseID in synapsesToRemove:
                        del self.synapses[synapseID]

    def reproduce(self, otherParent):
        child = Organism(self.config, self.inputSize, self.outputSize, self.rng)
        child.neurons = set(self.neurons)

        for synapseID, synapse in self.synapses.items():
            if synapseID in otherParent.synapses:
                chosen = synapse if self.rng.random() > 0.5 else otherParent.synapses[synapseID]
                child.synapses[synapseID] = copy.deepcopy(chosen)
            else:
                child.synapses[synapseID] = copy.deepcopy(synapse) # Assume self is fitter, so we take self's disjoint genes
        return child
