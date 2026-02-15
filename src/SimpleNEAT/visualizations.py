import numpy as np
import cv2 as cv

def createVisualization(observation, canvasStartPoint, canvasEndPoint, gridColor=(0.2, 0.2, 0.2, 1.0)):
    # TODO: Put this in config
    coloredObservation = True
    positiveColor = (0.0, 1.0, 1.0, 1.0)
    negativeColor = (1.0, 0.0, 0.0, 1.0)
    gridThickness = 5

    if observation.ndim == 1: observation = observation[... , np.newaxis, np.newaxis]
    if observation.ndim == 2: observation = observation[... , np.newaxis]
    observation = np.clip(observation, -1, 1)
    rows, cols, _ = observation.shape
    
    if coloredObservation:
        positiveMask = np.clip( observation, 0, 1)
        maskNegative = np.clip(-observation, 0, 1)
        observationRGB = (positiveMask * positiveColor[:3]) + (maskNegative * negativeColor[:3])
        observationRGBA01 = np.concatenate((observationRGB, np.ones_like(observationRGB[:, :, 0:1])), axis=-1)
    else:
        observation01 = ((observation + 1) / 2.0)
        observationRGBA01 = np.stack([observation01, observation01, observation01, np.ones_like(observation01)], axis=-1)
    observationRGBA = observationRGBA01

    canvasHeightRequested = abs(canvasEndPoint[1] - canvasStartPoint[1])
    cellSize = (canvasHeightRequested - gridThickness * (rows + 1)) // rows

    gridHeight = (rows * cellSize) + ((rows + 1) * gridThickness)
    gridWidth  = (cols * cellSize) + ((cols + 1) * gridThickness)
    
    observationGrid = np.full((gridHeight, gridWidth, 4), gridColor)

    for row in range(rows):
        y_top = gridThickness + row * (cellSize + gridThickness)
        for column in range(cols):
            x_left = gridThickness + column * (cellSize + gridThickness)
            observationGrid[y_top:y_top + cellSize, x_left:x_left + cellSize] = observationRGBA[row, column]

    return (observationGrid*255).astype(np.uint8)

def embedForegroundOnFrame(foreground, frame, position, globalAlpha=1.0):
    offsetX, offsetY = position
    foregroundHeight, foregroundWidth = foreground.shape[:2]

    backgroundToBeBlended = frame[offsetY:offsetY + foregroundHeight, offsetX:offsetX + foregroundWidth]
    pixelAlpha = foreground[:, :, 3:4] * globalAlpha

    blended = (backgroundToBeBlended.astype(np.float32) * (1.0 - pixelAlpha) + foreground[:, :, :3].astype(np.float32) * pixelAlpha).astype(np.uint8)
    frame[offsetY:offsetY + foregroundHeight, offsetX:offsetX + foregroundWidth] = blended
    return frame

def percentCoordsToIdx(point: tuple[float, float], canvas: np.ndarray) -> tuple[float, float]:
    x, y = int(point[0]*canvas.shape[1]), int(point[1]*canvas.shape[0])
    return (x, y)
