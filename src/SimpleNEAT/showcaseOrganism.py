import imageio
import numpy as np
from SimpleNEAT.utils import ensurePath
from SimpleNEAT.visualizations import createVisualization, embedForegroundOnFrame, percentCoordsToIdx

def showcaseOrganism(organism, environmentMaker, config):
    print(f"Starting recording the showcase video")
    environment = environmentMaker(render_mode="rgb_array")
    videoPath = ensurePath(config.folder, config.filename if config.filename.endswith(".mp4") else config.filename + ".mp4")
    frames = []
    for i in range(config.episodes):
        state = environment.reset()
        organism.clearMemory()
        done, score = False, 0
        while not done:
            action = organism(state)
            state, reward, done = environment.step(action)
            score += reward

            environmentFrame    = environment.render().repeat(config.upscalingFactor, axis=0).repeat(config.upscalingFactor, axis=1)
            visualization       = createVisualization(state, percentCoordsToIdx((0.1, 0.2), environmentFrame), percentCoordsToIdx((0.9, 0.6), environmentFrame))
            finalFrame          = embedForegroundOnFrame(visualization, environmentFrame, percentCoordsToIdx((0.1, 0.2), environmentFrame), 1.0)

            frames.append(finalFrame)
        print(f"Showcase Episode {i+1:>2}. Score: {score:>8.2f}")
    imageio.mimwrite(videoPath, frames, fps=config.fps)
