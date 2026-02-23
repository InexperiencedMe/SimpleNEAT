import imageio
import numpy as np
from SimpleNEAT.utils import ensurePath
from SimpleNEAT.visualizations import *

def showcaseOrganism(organism, solver, environmentMaker, config):
    print(f"Starting recording the showcase video")
    environment = environmentMaker(render_mode="rgb_array")
    videoPath = ensurePath(config.folder, config.filename if config.filename.endswith(".mp4") else config.filename + ".mp4")
    with imageio.get_writer(videoPath, fps=config.fps) as writer:
        for i in range(config.episodes):
            observation = environment.reset()
            organism.clearMemory()
            done, score = False, 0
            while not done:
                action = organism(observation)

                environmentFrame    = imgUint8ToFloat32(environment.render().repeat(config.upscalingFactor, axis=0).repeat(config.upscalingFactor, axis=1))
                visualizationHeight, visualizationWidth = percentCornersToHeightAndWidth(environmentFrame, config.vizCornerTopLeftInPercent, config.vizCornerBottomRightInPercent)
                visualization       = createVisualization(visualizationHeight, visualizationWidth, organism, solver, observation, action, config.visualization)
                finalFrame          = embedForegroundOnFrame(visualization, environmentFrame, percentCoordsToIdx(environmentFrame, config.vizCornerTopLeftInPercent), config.globalAlpha)
                writer.append_data(imgFloat32ToUint8(finalFrame))

                observation, reward, done = environment.step(action)
                score += reward

            print(f"Showcase Episode {i+1:>2}. Score: {score:>8.2f}")
