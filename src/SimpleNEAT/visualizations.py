import numpy as np

def createVisualization(observation, upscalingFactor):
    if observation.ndim == 1: observation = np.atleast_2d(observation)
    if observation.ndim == 2: observation = np.stack([observation] * 3, axis=-1)
    if observation.ndim == 3: observation = observation.repeat(upscalingFactor, axis=0).repeat(upscalingFactor, axis=1)
    return (observation * 255).astype(np.uint8)

def embedForegroundOnFrame(foreground, frame, posPercentY, posPercentX, alpha):
    offsetY, offsetX = int(posPercentY*frame.shape[0]), int(posPercentX*frame.shape[1])

    height, width, _ = foreground.shape
    background = frame[offsetY:offsetY + height, offsetX:offsetX + width]
    blended = (background * (1 - alpha) + foreground * alpha).astype(np.uint8)
    frame[offsetY:offsetY + height, offsetX:offsetX + width] = blended
    return frame
