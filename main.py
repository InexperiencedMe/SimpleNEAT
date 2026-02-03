import numpy as np
import gymnasium as gym
import argparse
import copy
from dataclasses import dataclass
from collections import defaultdict
from environmentUtils import CleanLunarLander

class InnovationTracker:
    def __init__(self, inputSize, outputSize):
        self.novelSynapses  = {}  # {(source, destination): novelSynapseID}
        self.novelNeurons   = {}  # {(source, destination): novelNeuronID}
        self.neuronCount    = inputSize + outputSize
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
    source:      int
    destination: int
    weight:      float
    enabled:     bool

class Organism:
    def __init__(self, config, inputSize, outputSize):
        self.config             = config
        self.inputSize          = inputSize
        self.inputSizeWithBias  = inputSize + 1
        self.biasNode           = inputSize
        self.outputSize         = outputSize
        self.neurons    = set(range(self.inputSizeWithBias + self.outputSize))
        self.synapses   = {} # {synapseID: Synapse}
        self.memory     = defaultdict(float)
        self.fitness    = 0.0 # NOTE: Aaaaaghhhhhhh nooooooooo, separation of conceeeeeernsss

        self.mutationChance_modifyWeight = config.mutationChanceModifyWeight
        self.mutationChance_newSynapse   = config.mutationChanceNewSynapse
        self.mutationChance_newNeuron    = config.mutationChanceNewNeuron
        self.weightMutationScale         = config.weightMutationScale
        self.resetWeightWhenMutatingIt   = config.resetWeightChance

    def clearMemory(self):
        self.memory.clear()

    def __call__(self, inputs):
        for i, input in enumerate(inputs):
            self.memory[i] = input
        self.memory[self.biasNode] = 1.0
        
        newState = defaultdict(float)
        for synapse in self.synapses.values():
            if synapse.enabled:
                newState[synapse.destination] += self.memory[synapse.source] * synapse.weight

        for neuronID, activatonInput in newState.items():
            self.memory[neuronID] = np.tanh(activatonInput)
        
        return np.array([self.memory[self.inputSizeWithBias + i] for i in range(self.outputSize)])
    
    def initializeSynapses(self, tracker):
        for input in range(self.inputSizeWithBias):
            for output in range(self.inputSizeWithBias, self.inputSizeWithBias + self.outputSize):
                self.synapses[tracker.getSynapseID(input, output)] = Synapse(input, output, rng.normal(0, 1.0), True)

    def mutate(self, tracker):
        if rng.random() < self.mutationChance_modifyWeight:
            for synapse in self.synapses.values():
                if rng.random() < self.resetWeightWhenMutatingIt:
                    synapse.weight = rng.normal(0, 1.0)
                else:
                    synapse.weight += rng.normal(0, self.weightMutationScale)

        if rng.random() < self.mutationChance_newSynapse:
            validSources        = list(self.neurons)
            validDestinations   = [n for n in self.neurons if n >= self.inputSizeWithBias]
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
        child = Organism(self.config, self.inputSize, self.outputSize)
        child.neurons = set(self.neurons)

        for synapseID, synapse in self.synapses.items():
            if synapseID in otherParent.synapses:
                chosen = synapse if rng.random() > 0.5 else otherParent.synapses[synapseID]
                child.synapses[synapseID] = copy.deepcopy(chosen)
            else:
                child.synapses[synapseID] = copy.deepcopy(synapse) # Assume self is fitter, so we take self's disjoint genes
        return child

class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members        = [representative]
        self.averageFitness = 0.0
        self.stagnation     = 0
        self.maxFitnessEver = -np.inf

class NEAT:
    def __init__(self, config, inputSize, outputSize):
        self.config         = config
        self.populationSize = config.populationSize
        self.inputSize      = inputSize
        self.outputSize     = outputSize
        self.compatibilityThreshold       = config.defaultCompatibilityThreshold
        self.compatibilityAdjustmentSpeed = config.compatibilityAdjustmentSpeed
        self.targetSpeciesCount           = config.populationSize // config.targetSpeciesSize
        self.survivalThreshold            = config.survivalThreshold
        self.stagnationThreshold          = config.stagnationThreshold
        self.lossWeight_E                 = config.lossWeightExcess
        self.lossWeight_D                 = config.lossWeightDisjoint
        self.lossWeight_W                 = config.lossWeightWeightsDifference
        self.tracker = InnovationTracker(inputSize, outputSize)
        self.species = []

    def getInitialPopulation(self):
        population = []
        for _ in range(self.populationSize):
            organism = Organism(self.config, self.inputSize, self.outputSize)
            organism.initializeSynapses(self.tracker)
            population.append(organism)
        return population                

    def speciate(self, population):
        for species in self.species:
            species.members = []

        for organism in population:
            placed = False
            for species in self.species:
                if self.calculateGeneticDistance(organism, species.representative) < self.compatibilityThreshold:
                    species.members.append(organism)
                    placed = True; break
            if not placed:
                self.species.append(Species(organism))

        self.species = [species for species in self.species if len(species.members) > 0]

    def calculateDynamicCompatibilityThreshold(self, speciesCount):
        if      speciesCount < self.targetSpeciesCount: self.compatibilityThreshold -= self.compatibilityAdjustmentSpeed
        elif    speciesCount > self.targetSpeciesCount: self.compatibilityThreshold += self.compatibilityAdjustmentSpeed
        return max(0.3, self.compatibilityThreshold)

    def getNewPopulation(self, population):
        self.speciate(population)
        self.compatibilityThreshold = self.calculateDynamicCompatibilityThreshold(len(self.species))

        # Sort, Update Stagnation, Find Global Best
        bestFitnessGlobal = -np.inf
        for species in self.species:
            species.members.sort(key=lambda organism: organism.fitness, reverse=True)
            bestInSpecies = species.members[0]
            species.representative = bestInSpecies

            if bestInSpecies.fitness > species.maxFitnessEver:
                species.stagnation = 0
                species.maxFitnessEver = bestInSpecies.fitness
            else:
                species.stagnation += 1
            
            if bestInSpecies.fitness > bestFitnessGlobal: 
                bestFitnessGlobal = bestInSpecies.fitness

        # Remove stagnant unless contains best global member
        nonstagnantSpecies = []
        for species in self.species:
            if species.stagnation < self.stagnationThreshold or species.members[0].fitness >= bestFitnessGlobal:
                nonstagnantSpecies.append(species)
        self.species = nonstagnantSpecies

        # Calculate Adjusted Fitness for nonstagnant species
        totalAdjustedFitness = 0
        newPopulation = []
        for species in self.species:
            minimumFitness = min(organism.fitness for organism in species.members)
            shift = abs(minimumFitness) if minimumFitness < 0 else 0

            speciesAdjustedFitnessSum = 0
            for organism in species.members:
                speciesAdjustedFitnessSum += (organism.fitness + shift) / len(species.members)
            
            species.averageFitness = speciesAdjustedFitnessSum / len(species.members)
            totalAdjustedFitness += species.averageFitness

            if len(species.members) >= self.targetSpeciesCount // 4:
                newPopulation.append(copy.deepcopy(species.members[0])) # Elitism

        while len(newPopulation) < self.populationSize:
            draw = rng.uniform(0, totalAdjustedFitness)
            currentThreshold = 0
            selectedSpecies = self.species[0]
            for species in self.species:
                currentThreshold += species.averageFitness
                if currentThreshold > draw:
                    selectedSpecies = species
                    break
            
            survivalCutoff = max(1, int(len(selectedSpecies.members) * self.survivalThreshold))
            pool = selectedSpecies.members[:survivalCutoff]
            parent1 = rng.choice(pool)
            parent2 = rng.choice(pool)

            child = parent1.reproduce(parent2) if parent1.fitness > parent2.fitness else parent2.reproduce(parent1)
            child.mutate(self.tracker)
            newPopulation.append(child)
        return newPopulation

    def calculateGeneticDistance(self, organism1, organism2):
        synapseIDs1, synapseIDs2 = set(organism1.synapses.keys()), set(organism2.synapses.keys())

        matching = synapseIDs1 & synapseIDs2
        if matching:
            weightsDifference = sum(abs(organism1.synapses[key].weight - organism2.synapses[key].weight) for key in matching) / len(matching)
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
        return (self.lossWeight_E*excessCount + self.lossWeight_D*disjointCount) / maxSynapses + self.lossWeight_W*weightsDifference
        

def main(args):
    env = CleanLunarLander(gym.make("LunarLanderContinuous-v3", render_mode=None))
    solver = NEAT(args, inputSize=env.observationSize, outputSize=env.actionSize)
    population = solver.getInitialPopulation()

    generation = 0
    maxFitnessEver = -np.inf
    while True:
        maxFitnessThisGeneration = -np.inf
        fitnessScoresThisGeneration = 0
        for organism in population:
            fitnessScore = 0
            for _ in range(args.evaluationEpisodes):
                state = env.reset(seed = rng.integers(0, 1000).item())
                organism.clearMemory()
                while True:
                    action = organism(state)
                    state, reward, done = env.step(action)
                    fitnessScore += reward
                    if done: break

            organism.fitness = fitnessScore / args.evaluationEpisodes
            fitnessScoresThisGeneration += organism.fitness

            if organism.fitness > maxFitnessEver:
                maxFitnessEver = organism.fitness
                bestOrganism = copy.deepcopy(organism)

            if organism.fitness > maxFitnessThisGeneration:
                maxFitnessThisGeneration = organism.fitness

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
    parser.add_argument("-n",  "--runName",                       type=str,     default="myNEATrun")
    parser.add_argument("-s",  "--seed",                          type=int,     default=123)
    parser.add_argument("-p",  "--populationSize",                type=int,     default=150)
    parser.add_argument("-t",  "--targetFitness",                 type=float,   default=320.0)
    parser.add_argument("-st", "--survivalThreshold",             type=float,   default=0.2)
    parser.add_argument("-sg", "--stagnationThreshold",           type=int,     default=50)
    parser.add_argument("-ee", "--evaluationEpisodes",            type=int,     default=3)
    parser.add_argument("-ct", "--defaultCompatibilityThreshold", type=float,   default=3.0)
    parser.add_argument("-cs", "--compatibilityAdjustmentSpeed",  type=float,   default=0.2)
    parser.add_argument("-ss", "--targetSpeciesSize",             type=int,     default=15)
    parser.add_argument("-le", "--lossWeightExcess",              type=float,   default=1.0)
    parser.add_argument("-ld", "--lossWeightDisjoint",            type=float,   default=1.0)
    parser.add_argument("-lw", "--lossWeightWeightsDifference",   type=float,   default=0.4)
    parser.add_argument("-mw", "--mutationChanceModifyWeight",    type=float,   default=0.5)
    parser.add_argument("-ms", "--mutationChanceNewSynapse",      type=float,   default=0.1)
    parser.add_argument("-mn", "--mutationChanceNewNeuron",       type=float,   default=0.05)
    parser.add_argument("-mr", "--resetWeightChance",             type=float,   default=0.1)
    parser.add_argument("-ws", "--weightMutationScale",           type=float,   default=0.01)

    args = parser.parse_args()
    
    seed = args.seed
    rng = np.random.default_rng(seed)

    main(args)