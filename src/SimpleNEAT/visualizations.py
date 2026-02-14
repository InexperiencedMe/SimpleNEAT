import numpy as np
import cv2 as cv

def createVisualization(observation, upscalingFactor, coloredObservation=True, positiveColor=(0, 255, 255), negativeColor=(255, 0, 0)):
    if observation.ndim == 1: observation = np.atleast_2d(observation).T
    observation = np.clip(observation, -1, 1)

    if coloredObservation:
        positiveMask = np.clip(observation, 0, 1)
        negativeMask = np.clip(-observation, 0, 1)
        channels = [(positiveMask * positiveColor[i]) + (negativeMask * negativeColor[i]) for i in range(3)]
        observation = np.stack(channels, axis=-1)
    else:
        observation = (observation + 1) / 2.0 * 255
        observation = np.stack([observation] * 3, axis=-1)

    observationVisualization = observation.astype(np.uint8).repeat(upscalingFactor, axis=0).repeat(upscalingFactor, axis=1)


    return observationVisualization

def embedForegroundOnFrame(foreground, frame, position, alpha):
    offsetX, offsetY = percentageCoordsToNumbers(position, frame)

    height, width, _ = foreground.shape
    background = frame[offsetY:offsetY + height, offsetX:offsetX + width]
    blended = (background * (1 - alpha) + foreground * alpha).astype(np.uint8)
    frame[offsetY:offsetY + height, offsetX:offsetX + width] = blended
    return frame

def percentageCoordsToNumbers(point: tuple[float, float], canvas: np.ndarray) -> tuple[float, float]:
    x, y = int(point[0]*canvas.shape[1]), int(point[1]*canvas.shape[0])
    return (x, y)
