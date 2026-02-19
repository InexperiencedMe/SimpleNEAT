import imageio
import numpy as np
from SimpleNEAT.utils import ensurePath
from SimpleNEAT.visualizations import *

def showcaseOrganism(organism, environmentMaker, config):
    print(f"Starting recording the showcase video")
    environment = environmentMaker(render_mode="rgb_array")
    videoPath = ensurePath(config.folder, config.filename if config.filename.endswith(".mp4") else config.filename + ".mp4")
    frames = []
    for i in range(config.episodes):
        observation = environment.reset()
        organism.clearMemory()
        done, score = False, 0
        while not done:
            action = organism(observation)

            environmentFrame    = imgUint8ToFloat32(environment.render().repeat(config.upscalingFactor, axis=0).repeat(config.upscalingFactor, axis=1))
            visualizationHeight, visualizationWidth = percentCornersToHeightAndWidth(environmentFrame, (0.1, 0.2), (0.9, 0.6))
            visualization       = createVisualization(visualizationHeight, visualizationWidth, organism, observation,  action)
            finalFrame          = embedForegroundOnFrame(visualization, environmentFrame, percentCoordsToIdx(environmentFrame, (0.1, 0.2)), 1.0)
            frames.append(imgFloat32ToUint8(finalFrame))

            observation, reward, done = environment.step(action)
            score += reward

        print(f"Showcase Episode {i+1:>2}. Score: {score:>8.2f}")
    imageio.mimwrite(videoPath, frames, fps=config.fps)
