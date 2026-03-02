import numpy as np
import copy
from SimpleNEAT.innovationTracker import InnovationTracker
from SimpleNEAT.organism import Organism
from SimpleNEAT.species import Species

class NEAT:
    def __init__(self, config, inputSize, outputSize, rng=None):
        self.config         = config
        self.populationSize = config.populationSize
        self.inputSize      = inputSize
        self.outputSize     = outputSize
        self.rng            = rng if rng is not None else np.random.default_rng(config.seed)
        self.tracker        = InnovationTracker(inputSize, outputSize)
        self.species        = []

        self.compatibilityThreshold       = config.defaultCompatibilityThreshold
        self.targetSpeciesCount           = config.populationSize // config.targetSpeciesSize

    def getInitialPopulation(self):
        population = []
        for _ in range(self.populationSize):
            organism = Organism(self.config, self.inputSize, self.outputSize, self.rng)
            if self.config.initializeSynapses: organism.initializeSynapses(self.tracker)
            population.append(organism)
        return population                

    def speciate(self, evaluatedPopulation): # A list of (Organism, fitness)
        for species in self.species:
            species.reset()

        for organism, fitness in evaluatedPopulation:
            placed = False
            for species in self.species:
                if self.calculateGeneticDistance(organism, species.representative) <= self.compatibilityThreshold:
                    species.addMember(organism, fitness)
                    placed = True; break
            if not placed:
                newSpecies = Species(representative=organism)
                newSpecies.addMember(organism, fitness)
                self.species.append(newSpecies)

        self.species = [species for species in self.species if len(species.members) > 0]

    def calculateDynamicCompatibilityThreshold(self, speciesCount):
        if      speciesCount < self.targetSpeciesCount: self.compatibilityThreshold -= self.config.compatibilityAdjustmentSpeed
        elif    speciesCount > self.targetSpeciesCount: self.compatibilityThreshold += self.config.compatibilityAdjustmentSpeed
        return max(0.3, self.compatibilityThreshold)

    def getNewPopulation(self, population, fitnessScores):
        self.speciate(list(zip(population, fitnessScores)))
        self.compatibilityThreshold = self.calculateDynamicCompatibilityThreshold(len(self.species))

        # Sort, Update Stagnation, Find Global Best
        bestFitnessGlobal = -np.inf
        for species in self.species:
            species.members.sort(key=lambda member: member[1], reverse=True)
            
            bestOrganismInSpecies, bestFitnessInSpecies = species.members[0]
            species.representative = bestOrganismInSpecies 

            if bestFitnessInSpecies > species.maxFitnessEver:
                species.stagnation = 0
                species.maxFitnessEver = bestFitnessInSpecies
            else:
                species.stagnation += 1
            
            if bestFitnessInSpecies > bestFitnessGlobal: 
                bestFitnessGlobal = bestFitnessInSpecies

        # Remove stagnant unless contains best global member
        nonstagnantSpecies = []
        for species in self.species:
            bestFitnessInSpecies = species.members[0][1]
            if species.stagnation < self.config.stagnationThreshold or bestFitnessInSpecies >= bestFitnessGlobal:
                nonstagnantSpecies.append(species)
        self.species = nonstagnantSpecies

        # Calculate Average Fitness for nonstagnant species
        for species in self.species:
            speciesSize = len(species.members)
            fitnessSum = sum(fitness for _, fitness in species.members)
            species.averageFitness = fitnessSum / speciesSize
            
        minAverageFitness = min(species.averageFitness for species in self.species)
        shift = -minAverageFitness if minAverageFitness < 0 else 0

        totalAdjustedFitness = 0
        newPopulation = []
        for species in self.species:
            speciesSize = len(species.members)
            
            species.averageFitness += shift
            totalAdjustedFitness += species.averageFitness

            if speciesSize >= self.config.targetSpeciesSize // 4:
                elites = species.members[:min(self.config.elitism, speciesSize)]
                newPopulation.extend(copy.deepcopy(organism) for organism, _ in elites)

        # Creating new generation
        while len(newPopulation) < self.populationSize:
            draw = self.rng.uniform(0, totalAdjustedFitness)
            currentThreshold = 0
            selectedSpecies = self.species[0]
            for species in self.species:
                currentThreshold += species.averageFitness
                if currentThreshold > draw:
                    selectedSpecies = species
                    break
            
            survivalCutoff = max(1, int(len(selectedSpecies.members) * self.config.survivalThreshold))
            pool = selectedSpecies.members[:survivalCutoff]
            
            (parent1, parent1fitness) = self.rng.choice(pool)
            (parent2, parent2fitness) = self.rng.choice(pool)

            child = parent1.reproduce(parent2) if parent1fitness > parent2fitness else parent2.reproduce(parent1)
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

        # maxSynapses = max(len(synapseIDs1), len(synapseIDs2), 1) # TODO Delete or keep
        return (self.config.lossWeight_E*excessCount + self.config.lossWeight_D*disjointCount)# / maxSynapses + self.config.lossWeight_W*weightsDifference
    