import numpy as np
import cv2 as cv

def preprocessValuesForGridVisualization(values):
    if values.ndim == 1: values = values[... , np.newaxis, np.newaxis]
    if values.ndim == 2: values = values[... , np.newaxis]
    values = np.clip(values, -1, 1)
    rows, cols, _ = values.shape

    return values, rows, cols

def createVisualizationGrid(values, rows, cols, cellSize, config):
    if config.coloredObservation:
        positiveMask = np.clip( values, 0, 1)
        maskNegative = np.clip(-values, 0, 1)
        observationRGB = (positiveMask * config.positiveColor[:3]) + (maskNegative * config.negativeColor[:3])
        observationRGBA01 = np.concatenate((observationRGB, np.ones_like(observationRGB[:, :, 0:1])), axis=-1)
    else:
        observation01 = ((values + 1) / 2.0)
        observationRGBA01 = np.stack([observation01, observation01, observation01, np.ones_like(observation01)], axis=-1)
    observationRGBA = observationRGBA01

    observationVisualizationHeight = (rows * cellSize) + ((rows + 1) * config.gridThickness)
    observationVisualizationWidth  = (cols * cellSize) + ((cols + 1) * config.gridThickness)
    
    observationVisualization = np.full((observationVisualizationHeight, observationVisualizationWidth, 4), config.gridColor, dtype=np.float32)

    grid = []
    offset = cellSize // 2 # To make grid from cell centers
    for row in range(rows):
        y_top = config.gridThickness + row * (cellSize + config.gridThickness)
        for column in range(cols):
            x_left = config.gridThickness + column * (cellSize + config.gridThickness)
            observationVisualization[y_top:y_top + cellSize, x_left:x_left + cellSize] = observationRGBA[row, column]
            grid.append((y_top + offset, x_left + offset))

    return observationVisualization, grid

def visualizeSynapses(canvas, organism, obsGrid, outputGrid, cellSize, config):
    # NOTE: Damn I will need the innovation tracker here to know what neurons are splitting what links and have consistent positioning
    pass

def createVisualization(canvasHeight, canvasWidth, organism, observation, action, config):
    # FIXME: observation and action can be taked from organism.memory, but it's 1D only.. Hmm.
    canvas = np.zeros((canvasHeight, canvasWidth, 4), dtype=np.float32)

    cleanObservation, obsRows, obsCols  = preprocessValuesForGridVisualization(observation)
    cleanAction, actionRows, actionCols = preprocessValuesForGridVisualization(action)
    cellSize = (canvasHeight - config.gridThickness * (obsRows + 1)) // obsRows

    obsViz, obsCoords = createVisualizationGrid(cleanObservation, obsRows, obsCols, cellSize, config)
    obsVizHeight, obsVizWidth = obsViz.shape[:2]
    obsVizOffsetX = 0
    obsVizOffsetY = (canvasHeight - obsVizHeight) // 2
    canvas[obsVizOffsetY:obsVizOffsetY+obsVizHeight, obsVizOffsetX:obsVizOffsetX+obsVizWidth] = obsViz

    outputViz, outputsCoords = createVisualizationGrid(cleanAction, actionRows, actionCols, cellSize, config)
    outputVizHeight, outputVizWidth = outputViz.shape[:2]
    outputVizOffsetX = canvasWidth - outputVizWidth
    outputVizOffsetY = (canvasHeight - outputVizHeight) // 2
    canvas[outputVizOffsetY:outputVizOffsetY+outputVizHeight, outputVizOffsetX:outputVizOffsetX+outputVizWidth] = outputViz

    obsGrid     = [(y + obsVizOffsetY,      x + obsVizOffsetX)      for y, x in obsGrid]    # Confirmed correct :)
    outputGrid  = [(y + outputVizOffsetY,   x + outputVizOffsetX)   for y, x in outputGrid] # Confirmed correct :)

    # for y, x in obsCoords + outputsCoords:
    #     cv.circle(canvas, (x, y), radius=10, color=(0.0, 0.0, 1.0, 1.0), thickness=-1)
        

    # canvas = visualizeSynapses(canvas, organism, obsCoords, outputsCoords, cellSize, config)

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
    