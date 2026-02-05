import numpy as np

class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members        = [] # Stores tuples (Organism, fitness)
        self.averageFitness = 0.0
        self.stagnation     = 0
        self.maxFitnessEver = -np.inf

    def addMember(self, organism, fitness):
        self.members.append((organism, fitness))

    def reset(self):
        self.members = []