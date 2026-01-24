import argparse
import numpy as np
import gymnasium as gym
from environmentUtils import CleanLunarLander

def main(args):
    env = CleanLunarLander(gym.make("LunarLanderContinuous-v3", render_mode=None))
    solver = NEAT(args.populationSize, inputSize=env.observationSize, outputSize=env.actionSize)
    population = solver.getInitialPopulation()

    fitnessScores = np.zeros(args.populationSize, dtype=np.float32)
    while True:
        for i, organism in enumerate(population):
            fitnessScore = 0
            for _ in range(args.evaluationEpisodes):
                state = env.reset()
                organism.clearMemory()
                while True:
                    action = organism(state)
                    state, reward, done = env.step(action)
                    fitnessScore += reward
                    if done: break
            fitnessScores[i] = fitnessScore/args.evaluationEpisodes
        
        endConditionMet = np.max(fitnessScores) >= args.targetFitness
        if endConditionMet:
            break
        else:
            population = solver.getNewPopulation(fitnessScores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",  "--runName",                 type=str,   default="myNEATrun")
    parser.add_argument("-p",  "--populationSize",          type=int,   default=150)
    parser.add_argument("-t",  "--targetFitness",           type=float, default=300.0)
    parser.add_argument("-ee", "--evaluationEpisodes",      type=int,   default=3)

    main(parser.parse_args())