import numpy as np
import cv2 as cv

def preprocessValuesForGridVisualization(values):
    if values.ndim == 1: values = values[... , np.newaxis, np.newaxis]
    if values.ndim == 2: values = values[... , np.newaxis]
    values = np.clip(values, -1, 1)
    rows, cols, _ = values.shape

    return values, rows, cols

def createVisualizationGrid(values, rows, cols, cellSize, colored, positiveColor, negativeColor, gridColor, gridThickness): # TODO: after colored should be in config.visualization.grid...
    if colored:
        positiveMask = np.clip( values, 0, 1)
        maskNegative = np.clip(-values, 0, 1)
        observationRGB = (positiveMask * positiveColor[:3]) + (maskNegative * negativeColor[:3])
        observationRGBA01 = np.concatenate((observationRGB, np.ones_like(observationRGB[:, :, 0:1])), axis=-1)
    else:
        observation01 = ((values + 1) / 2.0)
        observationRGBA01 = np.stack([observation01, observation01, observation01, np.ones_like(observation01)], axis=-1)
    observationRGBA = observationRGBA01

    observationVisualizationHeight = (rows * cellSize) + ((rows + 1) * gridThickness)
    observationVisualizationWidth  = (cols * cellSize) + ((cols + 1) * gridThickness)
    
    observationVisualization = np.full((observationVisualizationHeight, observationVisualizationWidth, 4), gridColor, dtype=np.float32)

    grid = []
    for row in range(rows):
        y_top = gridThickness + row * (cellSize + gridThickness)
        for column in range(cols):
            x_left = gridThickness + column * (cellSize + gridThickness)
            observationVisualization[y_top:y_top + cellSize, x_left:x_left + cellSize] = observationRGBA[row, column]
            grid.append((y_top, x_left))

    return observationVisualization, grid

def createVisualization(canvasHeight, canvasWidth, organism, observation, action):
    canvas = np.zeros((canvasHeight, canvasWidth, 4), dtype=np.float32)

    coloredObservation = True
    positiveColor = (0.0, 1.0, 1.0, 1.0)
    negativeColor = (1.0, 0.0, 0.0, 1.0)
    gridColor     = (0.2, 0.2, 0.2, 1.0)
    gridThickness = 5

    cleanObservation, obsRows, obsCols  = preprocessValuesForGridVisualization(observation)
    cleanAction, actionRows, actionCols = preprocessValuesForGridVisualization(action)
    cellSize = (canvasHeight - gridThickness * (obsRows + 1)) // obsRows

    obsViz, obsGrid = createVisualizationGrid(cleanObservation, obsRows, obsCols, cellSize, coloredObservation, positiveColor, negativeColor, gridColor, gridThickness)
    obsVizHeight, obsVizWidth = obsViz.shape[:2]
    obsVizOffsetX = 0
    obsVizOffsetY = (canvasHeight - obsVizHeight) // 2
    canvas[obsVizOffsetY:obsVizOffsetY+obsVizHeight, obsVizOffsetX:obsVizOffsetX+obsVizWidth] = obsViz

    outputViz, outputGrid = createVisualizationGrid(cleanAction, actionRows, actionCols, cellSize, coloredObservation, positiveColor, negativeColor, gridColor, gridThickness)
    outputVizHeight, outputVizWidth = outputViz.shape[:2]
    outputVizOffsetX = canvasWidth - outputVizWidth
    outputVizOffsetY = (canvasHeight - outputVizHeight) // 2
    canvas[outputVizOffsetY:outputVizOffsetY+outputVizHeight, outputVizOffsetX:outputVizOffsetX+outputVizWidth] = outputViz

    # canvas = visualizeSynapses(observation, organism, canvas, obsGrid, outputGrid)

    return canvas

def embedForegroundOnFrame(foreground01, frame, position, globalAlpha=1.0):
    offsetX, offsetY = position
    foregroundHeight, foregroundWidth = foreground01.shape[:2]

    backgroundToBeBlended = frame[offsetY:offsetY + foregroundHeight, offsetX:offsetX + foregroundWidth]
    pixelAlpha = foreground01[:, :, 3:4] * globalAlpha

    blended = backgroundToBeBlended * (1.0 - pixelAlpha) + foreground01[:, :, :3] * pixelAlpha
    frame[offsetY:offsetY + foregroundHeight, offsetX:offsetX + foregroundWidth] = blended
    return frame

def percentCoordsToIdx(canvas: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    x, y = int(point[0]*canvas.shape[1]), int(point[1]*canvas.shape[0])
    return (x, y)

def percentCornersToHeightAndWidth(canvas: np.ndarray, topLeft: tuple[float, float], bottomRight: tuple[float, float]):
    topLeftX, topLeftY          = percentCoordsToIdx(canvas, topLeft)
    bottomRightX, bottomRightY  = percentCoordsToIdx(canvas, bottomRight)

    canvasHeight    = abs(topLeftY - bottomRightY)
    canvasWidth     = abs(topLeftX - bottomRightX)

    return canvasHeight, canvasWidth

def imgUint8ToFloat32(img):
    return img.astype(np.float32) / 255

def imgFloat32ToUint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    