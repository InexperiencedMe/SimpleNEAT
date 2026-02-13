import numpy as np

def createVisualization(observation, upscalingFactor, coloredObservation=True, positiveColor=(0, 255, 255), negativeColor=(255, 0, 0)):
    observation = np.clip(np.atleast_2d(observation), -1, 1)

    if coloredObservation:
        positiveMask = np.clip(observation, 0, 1)
        negativeMask = np.clip(-observation, 0, 1)
        channels = [(positiveMask * positiveColor[i]) + (negativeMask * negativeColor[i]) for i in range(3)]
        observation = np.stack(channels, axis=-1)
    else:
        observation = (observation + 1) / 2.0 * 255
        observation = np.stack([observation] * 3, axis=-1)

    observation = observation.repeat(upscalingFactor, axis=0).repeat(upscalingFactor, axis=1)
    return observation.astype(np.uint8)

def embedForegroundOnFrame(foreground, frame, posPercentY, posPercentX, alpha):
    offsetY, offsetX = int(posPercentY*frame.shape[0]), int(posPercentX*frame.shape[1])

    height, width, _ = foreground.shape
    background = frame[offsetY:offsetY + height, offsetX:offsetX + width]
    blended = (background * (1 - alpha) + foreground * alpha).astype(np.uint8)
    frame[offsetY:offsetY + height, offsetX:offsetX + width] = blended
    return frame
