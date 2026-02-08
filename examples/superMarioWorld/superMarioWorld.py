import numpy as np
import stable_retro as retro
import argparse
import copy
import multiprocessing
import signal
import pygame as pg
from SimpleNEAT.NEAT import NEAT

# --- Environment Wrapper ---
class CleanMario:
    def __init__(self, env):
        self.env = env
        self.observationSize = 16 * 14 
        self.actionSize = 12 
        
        # Stagnation variables
        self.time_since_last_reward = 0
        self.current_total_reward = 0.0

    def process_obs(self, obs):
        # Grayscale (luminance) and downsample
        gray = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])
        downsampled = gray[::16, ::16] 
        return downsampled.flatten() / 255.0

    def reset(self, seed=None):
        # Handle gymnasium/retro return differences
        res = self.env.reset(seed=seed)
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs = res
            
        self.time_since_last_reward = 0
        self.current_total_reward = 0.0
        return self.process_obs(obs)

    def step(self, action):
        binary_action = [1 if x > 0 else 0 for x in action]
        obs, reward, terminated, truncated, info = self.env.step(binary_action)
        
        self.current_total_reward += reward

        # --- Reward Stagnation Check ---
        if reward > 0:
            self.time_since_last_reward = 0
        else:
            self.time_since_last_reward += 1
            
        # End episode if no reward for ~5 seconds
        is_stagnant = self.time_since_last_reward > 300
        
        done = terminated or truncated or is_stagnant
        return self.process_obs(obs), reward, done

    def close(self):
        self.env.close()
        
workerEnvironment = None

def initializeWorker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global workerEnvironment
    workerEnvironment = CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="YoshiIsland1", render_mode=None))

def evaluateOrganism(organism, seeds):
    global workerEnvironment
    rewardsSum = 0
    for seed in seeds:
        state = workerEnvironment.reset(seed=int(seed))
        organism.clearMemory()
        
        steps = 0
        while steps < 4000:
            action = organism(state)
            state, reward, done = workerEnvironment.step(action)
            rewardsSum += reward
            steps += 1
            if done: break
            
    return rewardsSum / len(seeds)

def main(args):
    rng = np.random.default_rng(args.seed)
    
    # Init temp env
    temporaryEnv = CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="YoshiIsland1", render_mode=None))
    solver = NEAT(args, inputSize=temporaryEnv.observationSize, outputSize=temporaryEnv.actionSize, rng=rng)
    temporaryEnv.close()
    
    population = solver.getInitialPopulation()

    generation = 0
    maxFitnessEver = -np.inf
    bestOrganism = None
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=initializeWorker) as pool:
        try:
            while True:
                seedsMatrix = rng.integers(0, 1000, size=(args.populationSize, args.evaluationEpisodes))

                result = pool.starmap_async(evaluateOrganism, zip(population, seedsMatrix))
                fitnessScores = result.get(timeout=999999) 
                
                maxFitnessThisGeneration = -np.inf
                for i, evaluatedFitness in enumerate(fitnessScores):
                    if evaluatedFitness > maxFitnessEver:
                        maxFitnessEver = evaluatedFitness
                        bestOrganism = copy.deepcopy(population[i])
                    if evaluatedFitness > maxFitnessThisGeneration:
                        maxFitnessThisGeneration = evaluatedFitness

                avgFitness = np.mean(fitnessScores)
                print(f"Gen {generation:4}: Best: {maxFitnessThisGeneration:>8.2f} | Avg: {avgFitness:8.2f} | Best Ever: {maxFitnessEver:8.2f}")

                if maxFitnessEver >= args.targetFitness:
                    print("Target fitness reached!")
                    break
                else:
                    population = solver.getNewPopulation(population, fitnessScores)
                    generation += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            pool.terminate()
            pool.join()
        finally:
            pool.close()

    # --- Visualization ---
    print("\nVisualizing Best Organism...")
    pg.init()
    win = pg.display.set_mode((768, 672))
    pg.display.set_caption("NEAT Mario Replay")
    clock = pg.time.Clock()
    
    env_internal = retro.make("SuperMarioWorld-Snes-v0", state="YoshiIsland1", render_mode="rgb_array")
    env = CleanMario(env_internal)
    
    try:
        for i in range(3):
            # IMPORTANT: Call env.reset() to reset the wrapper's stagnation timer
            state = env.reset()
            bestOrganism.clearMemory()
            fitnessScore = 0
            
            print(f"Starting Replay Episode {i+1}...")
            
            while True:
                # 1. Cap at 60 FPS
                clock.tick(60)
                
                # 2. Get the actual screen pixels for rendering
                raw_obs = env_internal.get_screen()
                surf = pg.surfarray.make_surface(raw_obs.swapaxes(0, 1))
                win.blit(pg.transform.scale(surf, (768, 672)), (0, 0))
                pg.display.flip()

                # 3. AI Step
                action = bestOrganism(state)
                state, reward, done = env.step(action)
                
                fitnessScore += reward
                
                # Handle Quit
                quit_replay = False
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        quit_replay = True
                
                if done or quit_replay: 
                    break
                
            print(f"Episode {i+1} Reward: {fitnessScore:.2f}")
    except Exception as e:
        print(f"Visualization error: {e}")
    finally:
        env.close()
        pg.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",  "--runName",                       type=str,     default="marioNEAT")
    parser.add_argument("-s",  "--seed",                          type=int,     default=123)
    parser.add_argument("-t",  "--targetFitness",                 type=float,   default=10000.0)
    parser.add_argument("-p",  "--populationSize",                type=int,     default=100)
    parser.add_argument("-ss", "--targetSpeciesSize",             type=int,     default=10)
    parser.add_argument("-st", "--survivalThreshold",             type=float,   default=0.2)
    parser.add_argument("-e",  "--elitism",                       type=int,     default=2)
    parser.add_argument("-sg", "--stagnationThreshold",           type=int,     default=30)
    parser.add_argument("-ee", "--evaluationEpisodes",            type=int,     default=1)
    parser.add_argument("-ct", "--defaultCompatibilityThreshold", type=float,   default=3.0)
    parser.add_argument("-cs", "--compatibilityAdjustmentSpeed",  type=float,   default=0.2)
    parser.add_argument("-le", "--lossWeightExcess",              type=float,   default=1.0)
    parser.add_argument("-ld", "--lossWeightDisjoint",            type=float,   default=1.0)
    parser.add_argument("-lw", "--lossWeightWeightsDifference",   type=float,   default=0.0)
    parser.add_argument("-mw", "--mutationChanceModifyWeight",    type=float,   default=0.8)
    parser.add_argument("-ms", "--mutationChanceNewSynapse",      type=float,   default=0.05)
    parser.add_argument("-mn", "--mutationChanceNewNeuron",       type=float,   default=0.03)
    parser.add_argument("-mr", "--resetWeightChance",             type=float,   default=0.1)
    parser.add_argument("-ws", "--weightMutationScale",           type=float,   default=0.01)
    main(parser.parse_args())