import numpy as np
import gymnasium as gym
import copy
from dataclasses import dataclass
from collections import defaultdict

# --- Configuration ---
INPUT_SIZE, OUTPUT_SIZE = 8, 2
POP_SIZE = 150
SPECIES_TARGET = 7
COMPAT_THRESH = 3.0
C1, C3 = 1.0, 0.4  # C2 is unused as discussed

# --- Globals ---
rng = np.random.default_rng(123)
novelSynapsesGlobal = {}  # {(source, dest): innovationID}
neuronCountGlobal   = INPUT_SIZE + OUTPUT_SIZE

@dataclass
class Synapse:
    sourceNeuron:       int
    destinationNeuron:  int
    weight:             float
    enabled:            bool

class Organism:
    def __init__(self):
        self.inputSize  = INPUT_SIZE
        self.outputSize = OUTPUT_SIZE
        self.neurons    = set(range(INPUT_SIZE + OUTPUT_SIZE))
        self.synapses   = {} # {innovationID: Synapse}
        self.fitness         = 0.0
        self.adjustedFitness = 0.0
        self.memory = defaultdict(float)
        
        # Probabilities
        self.mutationChance_modWeight   = 0.8
        self.mutationChance_newSynapse  = 0.1
        self.mutationChance_newNeuron   = 0.03

    def clearMemory(self):
        self.memory.clear()

    def __call__(self, inputs):
        # 1. Load Inputs into memory
        # Note: We copy memory to avoid modifying state during calculation of next step
        # effectively making this a synchronous update
        current_state = self.memory.copy()
        for i, val in enumerate(inputs):
            current_state[i] = val
        
        # 2. Propagate signals
        next_state_sums = defaultdict(float)
        for synapse in self.synapses.values():
            if synapse.enabled:
                # Signal comes from current_state (recurrence)
                # We do NOT apply tanh here. Tanh is an activation function applied to the node sum.
                signal = current_state[synapse.sourceNeuron]
                next_state_sums[synapse.destinationNeuron] += signal * synapse.weight

        # 3. Apply Activation (Tanh) and update memory
        for n in self.neurons:
            # Do not overwrite inputs (ids 0 to INPUT_SIZE-1)
            if n >= self.inputSize:
                self.memory[n] = np.tanh(next_state_sums[n])

        # 4. Return Outputs (The neurons immediately following inputs)
        return [self.memory[self.inputSize + i] for i in range(self.outputSize)]
    
    def mutate(self):
        # Weight Mutation
        if rng.random() < self.mutationChance_modWeight:
            for synapse in self.synapses.values():
                if rng.random() < 0.9:
                    synapse.weight += rng.normal(0, 0.2)
                else:
                    synapse.weight = rng.normal(0, 1.0)
        
        # Link Mutation
        if rng.random() < self.mutationChance_newSynapse:
            source = rng.choice(list(self.neurons))
            dest = rng.choice(list(self.neurons))
            
            # Logic: No connections TO inputs.
            # (Inputs are 0 to INPUT_SIZE-1)
            if dest >= self.inputSize and source != dest:
                key = (source, dest)
                existing = set((s.sourceNeuron, s.destinationNeuron) for s in self.synapses.values())
                
                if key not in existing:
                    if key not in novelSynapsesGlobal:
                        novelSynapsesGlobal[key] = len(novelSynapsesGlobal)
                    
                    innovID = novelSynapsesGlobal[key]
                    self.synapses[innovID] = Synapse(source, dest, rng.normal(0, 1.0), True)
        
        # Node Mutation
        if rng.random() < self.mutationChance_newNeuron and self.synapses:
            # Pick a random existing synapse to split
            innovID = rng.choice(list(self.synapses.keys()))
            synapse = self.synapses[innovID]
            
            if synapse.enabled:
                synapse.enabled = False
                
                global neuronCountGlobal
                newNeuron = neuronCountGlobal
                neuronCountGlobal += 1
                self.neurons.add(newNeuron)

                # Link 1: Source -> New (Weight = 1.0)
                k1 = (synapse.sourceNeuron, newNeuron)
                if k1 not in novelSynapsesGlobal: novelSynapsesGlobal[k1] = len(novelSynapsesGlobal)
                self.synapses[novelSynapsesGlobal[k1]] = Synapse(synapse.sourceNeuron, newNeuron, 1.0, True)

                # Link 2: New -> Dest (Weight = Old Weight)
                k2 = (newNeuron, synapse.destinationNeuron)
                if k2 not in novelSynapsesGlobal: novelSynapsesGlobal[k2] = len(novelSynapsesGlobal)
                self.synapses[novelSynapsesGlobal[k2]] = Synapse(newNeuron, synapse.destinationNeuron, synapse.weight, True)

    def distance(self, other):
        # Keys are the Innovation IDs
        k1, k2 = set(self.synapses.keys()), set(other.synapses.keys())
        
        disjointCount = len(k1 ^ k2)
        matching = k1 & k2
        
        if matching:
            weightDiff = sum(abs(self.synapses[k].weight - other.synapses[k].weight) for k in matching) / len(matching)
        else:
            weightDiff = 0.0
            
        return C1 * disjointCount + C3 * weightDiff

def crossover(parent1, parent2):
    # Assume parent1 is fitter (or equal)
    child = Organism()
    child.neurons = set(parent1.neurons) # Inherit topology nodes from fitter parent
    
    for innovID, synapse in parent1.synapses.items():
        # If both parents have this gene (matching)
        if innovID in parent2.synapses:
            chosen = synapse if rng.random() > 0.5 else parent2.synapses[innovID]
            # Must deepcopy because Synapse is a mutable dataclass
            child.synapses[innovID] = copy.deepcopy(chosen)
        else:
            # Disjoint/Excess from fitter parent are inherited
            child.synapses[innovID] = copy.deepcopy(synapse)
            
    return child

class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members = [representative]
        self.avgFitness = 0.0

def run_neat():
    global COMPAT_THRESH
    
    # 1. Initialize Population
    population = []
    for _ in range(POP_SIZE):
        org = Organism()
        # Create default Full-Mesh connections (Inputs -> Outputs)
        for i in range(INPUT_SIZE):
            for o in range(OUTPUT_SIZE):
                dest = INPUT_SIZE + o
                key = (i, dest)
                if key not in novelSynapsesGlobal: novelSynapsesGlobal[key] = len(novelSynapsesGlobal)
                org.synapses[novelSynapsesGlobal[key]] = Synapse(i, dest, rng.normal(0, 1.0), True)
        population.append(org)

    env = gym.make("LunarLanderContinuous-v3")
    
    for generation in range(200): # Shortened for demo
        # 2. Evaluate
        max_fitness = -10000
        best_org = None
        
        for org in population:
            total_reward = 0
            # Run 3 episodes per organism to reduce variance
            for _ in range(3):
                state, _ = env.reset()
                org.clearMemory()
                ep_r = 0
                while True:
                    # Action is raw output, Gym handles clipping usually, 
                    # but tanh implies [-1, 1], exactly what LunarLander wants
                    action = org(state) 
                    state, r, term, trunc, _ = env.step(np.array(action))
                    ep_r += r
                    if term or trunc: break
                total_reward += ep_r
            
            org.fitness = total_reward / 3.0
            if org.fitness > max_fitness:
                max_fitness = org.fitness
                best_org = org
        
        print(f"Gen {generation}: Best Fitness {max_fitness:.2f} | Species: {len(population)} (temp)")

        # 3. Speciation
        species_list = []
        for org in population:
            placed = False
            for s in species_list:
                if org.distance(s.representative) < COMPAT_THRESH:
                    s.members.append(org)
                    placed = True
                    break
            if not placed:
                species_list.append(Species(org))
        
        # Print Stats
        print(f"   -> Species Count: {len(species_list)} | Threshold: {COMPAT_THRESH:.2f}")
        
        if max_fitness >= 300: # Solved threshold
            print("SOLVED!")
            return best_org

        # 4. Reproduction
        new_population = []
        
        # Calculate Adjusted Fitness
        total_adjusted_fitness = 0
        valid_species = [s for s in species_list if len(s.members) > 0]
        
        for s in valid_species:
            # Sort by fitness (Descent)
            s.members.sort(key=lambda x: x.fitness, reverse=True)
            
            # Explicit Fitness Sharing
            # If fitness is negative, we need to handle it or NEAT breaks. 
            # Simple shift for probability calculation:
            min_fit = min(m.fitness for m in s.members)
            shift = 0 if min_fit > 0 else abs(min_fit) + 1.0
            
            avg_s = 0
            for m in s.members:
                m.adjustedFitness = (m.fitness + shift) / len(s.members)
                avg_s += m.adjustedFitness
            s.avgFitness = avg_s
            total_adjusted_fitness += s.avgFitness
            
            # Elitism: Add champion of each species directly
            new_population.append(copy.deepcopy(s.members[0]))

        # Fill the rest
        while len(new_population) < POP_SIZE:
            # Select Species based on roulette wheel
            pick = rng.uniform(0, total_adjusted_fitness)
            current = 0
            selected_species = valid_species[0]
            for s in valid_species:
                current += s.avgFitness
                if current > pick:
                    selected_species = s
                    break
            
            # Select 2 Parents from top 50% of species
            # If species only has 1 member, use it for both
            pool_size = max(1, len(selected_species.members) // 2)
            pool = selected_species.members[:pool_size]
            
            p1 = rng.choice(pool)
            p2 = rng.choice(pool)
            
            # Crossover (p1 must be best)
            if p1.fitness > p2.fitness:
                child = crossover(p1, p2)
            else:
                child = crossover(p2, p1)
            
            child.mutate()
            new_population.append(child)
            
        population = new_population

        # Dynamic Threshold Adjustment
        if len(species_list) < SPECIES_TARGET: COMPAT_THRESH -= 0.1
        elif len(species_list) > SPECIES_TARGET: COMPAT_THRESH += 0.1
        COMPAT_THRESH = max(0.3, COMPAT_THRESH)

    return best_org

if __name__ == "__main__":
    winner = run_neat()

    # Visualize 5 Episodes
    print("\nRunning 5 visualization episodes...")
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    
    for i in range(5):
        state, _ = env.reset()
        winner.clearMemory()
        total_r = 0
        while True:
            action = winner(state)
            state, r, term, trunc, _ = env.step(np.array(action))
            total_r += r
            if term or trunc: break
        print(f"Episode {i+1} Reward: {total_r:.2f}")
    
    env.close()