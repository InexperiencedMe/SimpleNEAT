import cv2 as cv
from SimpleNEAT.utils import ensurePath

def showcaseOrganism(organism, environmentMaker, config):
    environment = environmentMaker(render_mode="rgb_array")
    _ = environment.reset()
    frame = environment.render()
    frameHeight, frameWidth, _ = frame.shape
    targetHeight, targetWidth = frameHeight * config.upscalingFactor, frameWidth * config.upscalingFactor

    videoPath = ensurePath(config.folder, config.filename if config.filename.endswith(".mp4") else config.filename + ".mp4")
    videoWriter = cv.VideoWriter(videoPath, cv.VideoWriter_fourcc(*'mp4v'), config.fps, (targetWidth, targetHeight))
    for i in range(config.episodes):
        state = environment.reset()
        organism.clearMemory()
        done, score = False, 0
        while not done:
            frame = environment.render()
            
            resizedFrame = cv.resize(frame, (targetWidth, targetHeight), interpolation=cv.INTER_NEAREST)
            videoWriter.write(cv.cvtColor(resizedFrame, cv.COLOR_RGB2BGR))

            action = organism(state)
            state, reward, done = environment.step(action)
            score += reward
        print(f"Showcase Episode {i+1:<2}. Score: {score:>8.2f}")
    videoWriter.release()
