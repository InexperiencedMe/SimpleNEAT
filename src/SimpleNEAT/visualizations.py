import numpy as np
import cv2 as cv

def createVisualization(canvasHeight, canvasWidth, organism, observation, action):
    canvas = np.ones((canvasHeight, canvasWidth, 4))

    # NOTE: I will have to calculate cell size externally somehow...
    cleanObservation, obsRows, obsCols  = preprocessValuesForGridVisualization(observation)
    cleanAction, actionRows, actionCols = preprocessValuesForGridVisualization(action)
    cellSize = (canvasHeight - config.visualization.gridThickness * (obsRows + 1)) // obsRows

    obsViz, obsGrid = createObservationVisualization(cleanObservation, obsRows, obsCols, cellSize)
    obsVizHeight, obsVizWidth = obsViz.shape[:2]
    obsVizOffsetX = 0
    obsVizOffsetY = (canvasHeight - obsVizHeight) // 2
    canvas[obsVizOffsetY:obsVizOffsetY+obsVizHeight, obsVizOffsetX:obsVizOffsetX+obsVizWidth] = obsViz

    outputViz, outputGrid = createOutputVisualization(cleanAction, actionRows, actionCols, cellSize)
    outputVizHeight, outputVizWidth = outputViz.shape[:2]
    outputVizOffsetX = canvasWidth - outputVizWidth
    outputVizOffsetY = (canvasHeight - outputVizHeight) // 2
    canvas[outputVizOffsetY:outputVizOffsetY+outputVizHeight, outputVizOffsetX:outputVizOffsetX+outputVizWidth] = outputViz

    # canvas = visualizeSynapses(observation, organism, canvas, obsGrid, outputGrid)

    return canvas
    

# FIXME: This will be a subfunction to the main visualizer that puts 3 elements together: inputs, synapses, outputs
# def createVisualizationGrid(observation, availableHeight):
#     # TODO: Put this in config
#     coloredObservation = True
#     positiveColor = (0.0, 1.0, 1.0, 1.0)
#     negativeColor = (1.0, 0.0, 0.0, 1.0)
#     gridColor     = (0.2, 0.2, 0.2, 1.0)
#     gridThickness = 5

#     if observation.ndim == 1: observation = observation[... , np.newaxis, np.newaxis]
#     if observation.ndim == 2: observation = observation[... , np.newaxis]
#     observation = np.clip(observation, -1, 1)
#     rows, cols, _ = observation.shape
    
#     if coloredObservation:
#         positiveMask = np.clip( observation, 0, 1)
#         maskNegative = np.clip(-observation, 0, 1)
#         observationRGB = (positiveMask * positiveColor[:3]) + (maskNegative * negativeColor[:3])
#         observationRGBA01 = np.concatenate((observationRGB, np.ones_like(observationRGB[:, :, 0:1])), axis=-1)
#     else:
#         observation01 = ((observation + 1) / 2.0)
#         observationRGBA01 = np.stack([observation01, observation01, observation01, np.ones_like(observation01)], axis=-1)
#     observationRGBA = observationRGBA01

#     canvasHeight = abs(canvasEndPoint[1] - canvasStartPoint[1])
#     cellSize = (canvasHeight - gridThickness * (rows + 1)) // rows

#     observationVisualizationHeight = (rows * cellSize) + ((rows + 1) * gridThickness)
#     observationVisualizationWidth  = (cols * cellSize) + ((cols + 1) * gridThickness)
    
#     observationVisualization = np.full((observationVisualizationHeight, observationVisualizationWidth, 4), gridColor)

#     for row in range(rows):
#         y_top = gridThickness + row * (cellSize + gridThickness)
#         for column in range(cols):
#             x_left = gridThickness + column * (cellSize + gridThickness)
#             observationVisualization[y_top:y_top + cellSize, x_left:x_left + cellSize] = observationRGBA[row, column]

#     return (observationVisualization*255).astype(np.uint8)

def embedForegroundOnFrame(foreground, frame, position, globalAlpha=1.0):
    offsetX, offsetY = position
    foregroundHeight, foregroundWidth = foreground.shape[:2]

    backgroundToBeBlended = frame[offsetY:offsetY + foregroundHeight, offsetX:offsetX + foregroundWidth]
    pixelAlpha = foreground[:, :, 3:4] * globalAlpha

    blended = (backgroundToBeBlended.astype(np.float32) * (1.0 - pixelAlpha) + foreground[:, :, :3].astype(np.float32) * pixelAlpha).astype(np.uint8)
    frame[offsetY:offsetY + foregroundHeight, offsetX:offsetX + foregroundWidth] = blended
    return frame

def percentCoordsToIdx(canvas: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    x, y = int(point[0]*canvas.shape[1]), int(point[1]*canvas.shape[0])
    return (x, y)

def percentCornersToHeightAndWidth(canvas: np.ndarray, topLeft: tuple[float, float], bottomRight: tuple[float, float]):
    topLeftX, topLeftY          = percentCoordsToIdx(canvas, topLeft)
    bottomRightX, bottomRightY  = percentCoordsToIdx(canvas, bottomRight)

    canvasHeight    = abs(topLeftX - bottomRightX)
    canvasWidth     = abs(topLeftY - bottomRightY)

    return canvasHeight, canvasWidth
    