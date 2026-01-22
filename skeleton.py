import argparse
import numpy as np

def main(args):
    initializeEnvironment()

    fitnessScores = np.zeros(args.populationSize)
    endConditionMet = False
    while not endConditionMet:
        population = NEAT.getNewPopulation()
        endConditionMet = NEAT.evaluatePopulation()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--runName", type=str, default="myNEATrun")

    main(parser.parse_args())