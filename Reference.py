import numpy as np
import gymnasium as gym
import random
import copy
from collections import defaultdict

INPUTS, OUTPUTS     = 8, 2
POP_SIZE            = 150
SPECIES_TARGET      = 7
COMPAT_THRESH       = 3.0
MUTATE_WEIGHT, MUTATE_LINK, MUTATE_NODE = 0.8, 0.1, 0.03
C1, C2, C3 = 1.0, 1.0, 0.4
EPISODES_PER_EVALUATION = 3

innovationsGlobal   = {} # Global Innovation Tracker: {(in, out): innovation_id}
nodeCountGlobal     = INPUTS + OUTPUTS

class Genome:
    def __init__(self):
        self.genes = {}  # {innovationID: [in, out, weight, enabled]} # TODO: Rename to synapses and make Synapse class with source and destination
        self.nodes = set(range(INPUTS + OUTPUTS))
        self.fitness = 0.0
        self.fitnessAdjusted = 0.0
        self.memory = defaultdict(float)

    def resetMemory(self):
        self.memory.clear()

    def __call__(self, inputs):
        state = self.memory.copy()
        for i, input in enumerate(inputs):
            state[i] = input
        
        nextState = defaultdict(float)
        for gene in self.genes.values():
            if gene[3]:
                sourceNode, destinationNode, weight = gene[0], gene[1], gene[2]
                signal = state[sourceNode] if sourceNode < INPUTS else self.memory[sourceNode] 
                nextState[destinationNode] += signal * weight
        
        for n in self.nodes:
            if n >= INPUTS: 
                self.memory[n] = np.tanh(nextState[n])
        
        return [self.memory[INPUTS + i] for i in range(OUTPUTS)]

    def mutate(self):
        # Weights
        if random.random() < MUTATE_WEIGHT:
            for g in self.genes.values():
                if random.random() < 0.9: g[2] += np.random.normal(0, 0.2)
                else: g[2] = np.random.normal(0, 1.0)

        # Links
        if random.random() < MUTATE_LINK:
            i, o = random.choice(list(self.nodes)), random.choice(list(self.nodes))
            if i != o and o >= INPUTS: # No self-loop on inputs, no connection to inputs
                key = (i, o)
                existing = set((g[0], g[1]) for g in self.genes.values())
                if key not in existing:
                    if key not in innovationsGlobal: innovationsGlobal[key] = len(innovationsGlobal)
                    self.genes[innovationsGlobal[key]] = [i, o, np.random.normal(0, 1.0), True]

        # Nodes
        if random.random() < MUTATE_NODE and self.genes:
            g_idx = random.choice(list(self.genes.keys()))
            gene = self.genes[g_idx]
            if gene[3]:
                gene[3] = False
                global nodeCountGlobal
                new_node = nodeCountGlobal
                nodeCountGlobal += 1
                self.nodes.add(new_node)
                
                # In -> New
                k1 = (gene[0], new_node)
                if k1 not in innovationsGlobal: innovationsGlobal[k1] = len(innovationsGlobal)
                self.genes[innovationsGlobal[k1]] = [gene[0], new_node, 1.0, True]
                
                # New -> Out
                k2 = (new_node, gene[1])
                if k2 not in innovationsGlobal: innovationsGlobal[k2] = len(innovationsGlobal)
                self.genes[innovationsGlobal[k2]] = [new_node, gene[1], gene[2], True]

    def distance(self, other):
        k1, k2 = set(self.genes.keys()), set(other.genes.keys())
        disjoint = len(k1 ^ k2)
        matching = k1 & k2
        w_diff = sum(abs(self.genes[k][2] - other.genes[k][2]) for k in matching) / len(matching) if matching else 0
        return C1 * disjoint + C3 * w_diff

def crossover(p1, p2):
    # p1 is fitter
    child = Genome()
    child.nodes = set(p1.nodes)
    for innov, gene in p1.genes.items():
        if innov in p2.genes:
            child.genes[innov] = copy.deepcopy(gene if random.random() > 0.5 else p2.genes[innov])
        else:
            child.genes[innov] = copy.deepcopy(gene)
    return child

class Species:
    def __init__(self, rep):
        self.representative = rep
        self.members = [rep]
        self.avg_fitness = 0

def run_neat():
    global COMPAT_THRESH
    
    # 1. Init Pop
    pop = []
    for _ in range(POP_SIZE):
        g = Genome()
        for i in range(INPUTS):
            for o in range(OUTPUTS):
                k = (i, INPUTS + o)
                if k not in innovationsGlobal: innovationsGlobal[k] = len(innovationsGlobal)
                g.genes[innovationsGlobal[k]] = [i, INPUTS + o, np.random.normal(0, 1.0), True]
        pop.append(g)

    env = gym.make("LunarLanderContinuous-v3")
    
    for generation in range(1000):
        # 2. Evaluate (Average of 3 runs)
        max_fit = -10000
        total_pop_fit = 0
        best_g = None
        
        for g in pop:
            run_rewards = []
            for _ in range(EPISODES_PER_EVALUATION):
                state, _ = env.reset()
                g.resetMemory() # Clear recurrent state between episodes
                ep_r = 0
                while True:
                    action = g(state)
                    state, r, term, trunc, _ = env.step(np.array(action))
                    ep_r += r
                    if term or trunc: break
                run_rewards.append(ep_r)
            
            g.fitness = sum(run_rewards) / len(run_rewards)
            total_pop_fit += g.fitness
            
            if g.fitness > max_fit:
                max_fit = g.fitness
                best_g = g

        # 3. Speciation
        species = []
        for g in pop:
            placed = False
            for s in species:
                if g.distance(s.representative) < COMPAT_THRESH:
                    s.members.append(g)
                    placed = True
                    break
            if not placed: species.append(Species(g))

        # Stats
        avg_pop_fit = total_pop_fit / len(pop)
        print(f"Gen {generation}: Max Reward {max_fit:.2f} | Avg Pop Fit {avg_pop_fit:.2f} | Species: {len(species)} | Thresh: {COMPAT_THRESH:.2f}")

        if max_fit >= 300:
            print(f"SOLVED with avg fitness {max_fit:.2f} over {EPISODES_PER_EVALUATION} episodes!")
            return best_g

        # 4. Reproduction
        new_pop = []
        species = [s for s in species if s.members]
        
        # Adjust Fitness & Calculate Species Avg
        for s in species:
            s.members.sort(key=lambda x: x.fitness, reverse=True)
            adj_sum = 0
            for g in s.members:
                g.adjusted_fitness = g.fitness / len(s.members)
                adj_sum += g.adjusted_fitness
            s.avg_fitness = adj_sum 

        total_adj_fitness = sum(s.avg_fitness for s in species)
        
        # Elitism
        for s in species:
            new_pop.append(copy.deepcopy(s.members[0]))
        
        # Fill Pop
        while len(new_pop) < POP_SIZE:
            pick = random.uniform(0, total_adj_fitness)
            current = 0
            selected_s = species[0]
            for s in species:
                current += s.avg_fitness
                if current > pick:
                    selected_s = s
                    break
            
            # Tournament / Top 50%
            pool = selected_s.members[:max(1, len(selected_s.members)//2)]
            p1 = random.choice(pool)
            p2 = random.choice(pool)
            
            child = crossover(p1, p2)
            child.mutate()
            new_pop.append(child)
            
        pop = new_pop
        
        # Dynamic Threshold
        if len(species) < SPECIES_TARGET: COMPAT_THRESH -= 0.1
        elif len(species) > SPECIES_TARGET: COMPAT_THRESH += 0.1
        if COMPAT_THRESH < 0.3: COMPAT_THRESH = 0.3

    return best_g

if __name__ == "__main__":
    winner = run_neat()
    
    # Visualize 5 Episodes
    print("\nRunning 5 visualization episodes...")
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    
    for i in range(5):
        state, _ = env.reset()
        winner.resetMemory()
        total_r = 0
        while True:
            action = winner(state)
            state, r, term, trunc, _ = env.step(np.array(action))
            total_r += r
            if term or trunc: break
        print(f"Episode {i+1} Reward: {total_r:.2f}")
    
    env.close()