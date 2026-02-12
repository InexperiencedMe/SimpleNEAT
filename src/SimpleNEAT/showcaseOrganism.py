import imageio
import numpy as np
from SimpleNEAT.utils import ensurePath

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

            renderedFrame = environment.render().repeat(config.upscalingFactor, axis=0).repeat(config.upscalingFactor, axis=1)
            observationView = (state * 255).astype(np.uint8)
            
            if observationView.ndim == 2: observationView = np.stack([observationView] * 3, axis=-1)
            if observationView.ndim == 3: observationView = observationView.repeat(config.upscalingFactor*5, axis=0).repeat(config.upscalingFactor*5, axis=1)

            observationViewPositionY, observationViewPositionX = int(0.2*renderedFrame.shape[0]), int(0.1*renderedFrame.shape[1])

            h, w, _ = observationView.shape
            foreground = observationView[:h, :w].astype(float)
            background = renderedFrame[observationViewPositionY:observationViewPositionY+h, observationViewPositionX:observationViewPositionX+w].astype(float)
            blended = (background * (1 - 0.9)) + (foreground * 0.9)
            renderedFrame[observationViewPositionY:observationViewPositionY+h, observationViewPositionX:observationViewPositionX+w] = blended.astype(np.uint8)

            frames.append(renderedFrame)
        print(f"Showcase Episode {i+1:>2}. Score: {score:>8.2f}")
    imageio.mimwrite(videoPath, frames, fps=config.fps, macro_block_size=1)
