import cv2 as cv

def showcaseOrganism(organism, environmentMaker, episodes=10, fps=60, upscalingFactor=4):
    environment = environmentMaker(render_mode="rgb_array")
    state = environment.reset()
    frame = environment.render()
    frameHeight, frameWidth, _ = frame.shape
    targetHeight, targetWidth = frameHeight * upscalingFactor, frameWidth * upscalingFactor

    videoWriter = cv.VideoWriter("showcase.mp4", cv.VideoWriter_fourcc(*'mp4v'), fps, (targetWidth, targetHeight))
    for i in range(episodes):
        state = environment.reset()
        organism.clearMemory()
        done, score = False, 0
        while not done:
            frame = environment.render()
            
            resizedFrame = cv.resize(frame, (targetWidth, targetHeight), interpolation=cv.INTER_LINEAR)
            videoWriter.write(cv.cvtColor(resizedFrame, cv.COLOR_RGB2BGR))

            action = organism(state)
            step = environment.step(action)
            state, reward, done = step[0], step[1], (step[2] or step[3] if len(step) > 4 else step[2])
            score += reward
            
        print(f"Showcase Episode {i+1:2}. Score: {score:>8.2f}")
    
    if videoWriter: videoWriter.release()
    environment.close()