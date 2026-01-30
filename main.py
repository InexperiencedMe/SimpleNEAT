import argparse
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import gymnasium as gym
from environmentUtils import CleanLunarLander
import copy
rng = np.random.default_rng(123)

class InnovationTracker:
    def __init__(self, inputSize, outputSize):
        self.novelSynapses  = {}  # {(source, destination): novelSynapseID}
        self.novelNeurons   = {}  # {(source, destination): novelNeuronID}
        self.neuronCount    = (inputSize + 1) + outputSize # inputs + bias node + outputs
        self.synapseCount   = 0

    def getSynapseID(self, source, destination):
        link = (source, destination)
        if link not in self.novelSynapses:
            self.novelSynapses[link] = self.synapseCount
            self.synapseCount += 1
        return self.novelSynapses[link]
    
    def getNeuronID(self, source, destination):
        splitKey = (source, destination)
        if splitKey not in self.novelNeurons:
            self.novelNeurons[splitKey] = self.neuronCount
            self.neuronCount += 1
        return self.novelNeurons[splitKey]

@dataclass(slots=True)
class Synapse:
    source:             int
    destination:        int
    weight:             float
    enabled:            bool

class Organism:
    def __init__(self, inputSize, outputSize):
        self.inputSize  = inputSize + 1 # Accounting for bias node. I want it to be internal only
        self.outputSize = outputSize
        self.neurons    = set(range(self.inputSize + self.outputSize))
        self.synapses   = {} # {synapseID: Synapse}
        self.memory     = defaultdict(float)
        self.fitness         = 0.0 # NOTE: Aaaaaghhhhhhh nooooooooo, separation of conceeeeeernsss
        self.adjustedFitness = 0.0

        self.mutationChance_modifyWeight = 0.8
        self.mutationChance_newSynapse   = 0.1
        self.mutationChance_newNeuron    = 0.03

    def clearMemory(self):
        self.memory.clear()

    def __call__(self, inputs):
        inputs = np.append(inputs, 1.0) # Adding bias node
        for i, input in enumerate(inputs):
            self.memory[i] = input
        
        newState = defaultdict(float)
        for synapse in self.synapses.values():
            if synapse.enabled:
                newState[synapse.destination] += self.memory[synapse.source] * synapse.weight

        for neuron in self.neurons:
            if neuron >= self.inputSize:
                self.memory[neuron] = np.tanh(newState[neuron])
        
        return np.array([self.memory[self.inputSize + i] for i in range(self.outputSize)])
    
    def initializeSynapses(self, tracker):
        for input in range(self.inputSize):
            for output in range(self.inputSize, self.inputSize + self.outputSize):
                self.synapses[tracker.getSynapseID(input, output)] = Synapse(input, output, rng.normal(0, 1.0), True)

    def mutate(self, tracker):
        if rng.random() < self.mutationChance_modifyWeight:
            for synapse in self.synapses.values():
                if rng.random() < 0.9:
                    synapse.weight += rng.normal(0, 0.1)
                else:
                    synapse.weight = rng.normal(0, 1.0)

        if rng.random() < self.mutationChance_newSynapse:   
            validSources        = list(self.neurons)
            validDestinations   = [n for n in self.neurons if n >= self.inputSize]
            existingLinks       = set((synapse.source, synapse.destination) for synapse in self.synapses.values())
            for _ in range(10): # Retrying is faster than listing possible new links
                source        = int(rng.choice(validSources))
                destination   = int(rng.choice(validDestinations))
                link = (source, destination)
                if link not in existingLinks:
                    self.synapses[tracker.getSynapseID(source, destination)] = Synapse(source, destination, rng.normal(0, 1.0), True)
                    break
        
        if rng.random() < self.mutationChance_newNeuron and self.synapses:
            synapseKeys = list(self.synapses.keys())
            for _ in range(10):
                synapseToSplit = self.synapses[rng.choice(synapseKeys)]
                if synapseToSplit.enabled:
                    synapseToSplit.enabled = False

                    newNeuron = tracker.getNeuronID(synapseToSplit.source, synapseToSplit.destination)
                    self.neurons.add(newNeuron)

                    newLinkID1 = tracker.getSynapseID(synapseToSplit.source, newNeuron)
                    self.synapses[newLinkID1] = Synapse(synapseToSplit.source, newNeuron, 1.0, True)

                    newLinkID2 = tracker.getSynapseID(newNeuron, synapseToSplit.destination)
                    self.synapses[newLinkID2] = Synapse(newNeuron, synapseToSplit.destination, synapseToSplit.weight, True)
                    break

    def reproduce(self, otherParent):
        child = Organism(self.inputSize - 1, self.outputSize)
        child.neurons = set(self.neurons)

        for synapseID, synapse in self.synapses.items():
            if synapseID in otherParent.synapses:
                chosen = synapse if rng.random() > 0.5 else otherParent.synapses[synapseID]
                child.synapses[synapseID] = copy.deepcopy(chosen)
            else:
                child.synapses[synapseID] = copy.deepcopy(synapse) # Assume self is fitter, so we take their disjoint genes
        return child

class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members = [representative]
        self.averageFitness = 0.0
        # TODO: Add age plus best member?

class NEAT:
    def __init__(self, populationSize, inputSize, outputSize):
        self.populationSize = populationSize
        self.inputSize      = inputSize
        self.outputSize     = outputSize
        self.tracker        = InnovationTracker(inputSize, outputSize)
        self.compatibilityThreshold = 3.0 # TODO: Rethink positioning
        self.targetSpeciesCount     = 10


    def getInitialPopulation(self):
        population = []
        for _ in range(self.populationSize):
            organism = Organism(self.inputSize, self.outputSize)
            organism.initializeSynapses(self.tracker)
            population.append(organism)
        return population                

    def getNewPopulation(self, population):
        species = self.speciate(population) # TODO: We will want to preserve species across generations

        if len(species) < self.targetSpeciesCount: self.compatibilityThreshold -= 0.1
        elif len(species) > self.targetSpeciesCount: self.compatibilityThreshold += 0.1
        self.compatibilityThreshold = max(0.3, self.compatibilityThreshold)

        newPopulation = []
        totalAdjustedFitness = 0
        validSpecies = [s for s in species if len(s.members) > 0] # Useless for now, but maybe later when we preserve across gen?

        for species in validSpecies:
            species.members.sort(key=lambda organism: organism.fitness, reverse=True)
            minimumFitness = min(organism.fitness for organism in species.members)
            shift = 0 if minimumFitness > 0 else abs(minimumFitness)

            averageSpeciesFitness = 0
            for organism in species.members:
                organism.adjustedFitness = (organism.fitness + shift) / len(species.members)
                averageSpeciesFitness += organism.adjustedFitness
            species.averageFitness = averageSpeciesFitness
            totalAdjustedFitness += species.averageFitness

            newPopulation.append(copy.deepcopy(species.members[0])) # Elitism

        while len(newPopulation) < self.populationSize:
            draw = rng.uniform(0, totalAdjustedFitness)
            currentThreshold = 0
            selectedSpecies = validSpecies[0]
            for species in validSpecies:
                currentThreshold += species.averageFitness
                if currentThreshold > draw:
                    selectedSpecies = species
                    break
            
            pool = selectedSpecies.members[:max(1, len(selectedSpecies.members) // 2)] # Top 50%
            parent1 = rng.choice(pool)
            parent2 = rng.choice(pool)

            child = parent1.reproduce(parent2) if parent1.fitness > parent2.fitness else parent2.reproduce(parent1)
            child.mutate(self.tracker)
            newPopulation.append(child)
        return newPopulation


    def speciate(self, population): # TODO: I'd prefer if species remained, not got erased every generation
        species = []
        for organism in population:
            placed = False
            for specimen in species:
                if self.calculateGeneticDistance(organism, specimen.representative) < self.compatibilityThreshold:
                    specimen.members.append(organism)
                    placed = True; break
            if not placed:
                species.append(Species(organism))
        return species

    def calculateGeneticDistance(self, firstOrganism, secondOrganism):
        synapseIDs1, synapseIDs2 = set(firstOrganism.synapses.keys()), set(secondOrganism.synapses.keys())

        matchingSynapses = synapseIDs1 & synapseIDs2
        if matchingSynapses:
            weightsDifference = sum(abs(firstOrganism.synapses[key].weight - secondOrganism.synapses[key].weight) for key in matchingSynapses) / len(matchingSynapses)
        else:
            weightsDifference = 0.0

        disjointSynapseIDs = synapseIDs1 ^ synapseIDs2
        lowerMaxSynapseID = min(max(synapseIDs1) if synapseIDs1 else 0, max(synapseIDs2) if synapseIDs2 else 0)

        disjointCount, excessCount = 0, 0
        for synapseID in disjointSynapseIDs:
            if synapseID <= lowerMaxSynapseID:
                disjointCount += 1
            else:
                excessCount += 1

        maxSynapses = max(len(synapseIDs1), len(synapseIDs2))
        c1 = 1 # FIXME not harcoding
        c2 = 1
        c3 = 0.4
        return (c1*excessCount + c2*disjointCount)/maxSynapses + c3*weightsDifference
        

def main(args):
    env = CleanLunarLander(gym.make("LunarLanderContinuous-v3", render_mode=None))
    solver = NEAT(args.populationSize, inputSize=env.observationSize, outputSize=env.actionSize)
    population = solver.getInitialPopulation()

    generation = 0
    maxFitnessEver = -np.inf
    while True:
        maxFitnessThisGeneration = -np.inf
        fitnessScoresThisGeneration = 0
        for organism in population:
            fitnessScore = 0
            for _ in range(args.evaluationEpisodes):
                state = env.reset()
                organism.clearMemory()
                while True:
                    action = organism(state)
                    state, reward, done = env.step(action)
                    fitnessScore += reward
                    if done: break
            organism.fitness = fitnessScore / args.evaluationEpisodes
            if organism.fitness > maxFitnessEver: maxFitnessEver = organism.fitness; bestOrganism = organism
            if organism.fitness > maxFitnessThisGeneration: maxFitnessThisGeneration = organism.fitness
            fitnessScoresThisGeneration += organism.fitness
        fitnessScoresThisGeneration /= args.populationSize
        print(f"Generation {generation:4}: Best This Generation: {maxFitnessThisGeneration:>8.2f} | Average This Generation: {fitnessScoresThisGeneration:>8.2f} | Best Overall: {maxFitnessEver:>8.2f}")

        endConditionMet = np.max(maxFitnessEver) >= args.targetFitness
        if endConditionMet:
            break
        else:
            population = solver.getNewPopulation(population)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",  "--runName",                 type=str,   default="myNEATrun")
    parser.add_argument("-p",  "--populationSize",          type=int,   default=150)
    parser.add_argument("-t",  "--targetFitness",           type=float, default=300.0)
    parser.add_argument("-ee", "--evaluationEpisodes",      type=int,   default=3)

    main(parser.parse_args())