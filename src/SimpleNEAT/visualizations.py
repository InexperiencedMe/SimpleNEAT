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
        # FIXME: Not good. If observation ranges from 0 to 1, we make the range 0.5 - 1.0 which is terrible. For outputs fine, for inputs nahh
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

def translateNeuronToCoords(neuron, organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize):
    # TODO: A vast improvement would be calling this once for the organism and then indexing calculating neuron placements for all episodes and steps
    if neuron < organism.inputSize:
        return obsCoords[neuron]
    
    if neuron == organism.biasNeuron:
        lastObsCellY, lastObsCellX = obsCoords[-1]
        return lastObsCellY, lastObsCellX + cellSize*4
    
    if neuron < organism.inputSizeWithBias + organism.outputSize:
        return outputsCoords[neuron - organism.inputSizeWithBias]
    
    source, destination = neuronToLinkMap[neuron]
    sourceY, sourceX            = translateNeuronToCoords(source,       organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize)
    destinationY, destinationX  = translateNeuronToCoords(destination,  organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize)

    return (sourceY + destinationY) // 2, (sourceX + destinationX) // 2

def visualizeSynapses(canvas, organism, solver, obsCoords, outputsCoords, cellSize, config):
    canvas = imgFloat32ToUint8(canvas) # cv2 antialiasting works only on uint8 :|
    neuronToLinkMap = {v: k for k, v in solver.tracker.novelNeurons.items()}
    for synapse in organism.synapses.values():
        startpointY, startpointX    = translateNeuronToCoords(synapse.source,       organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize)
        endpointY, endpointX        = translateNeuronToCoords(synapse.destination,  organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize)

        arrowLength = np.sqrt((startpointY - endpointY)**2 + (startpointX - endpointX)**2)
        if arrowLength != 0: # TODO: We could potentially display self recursion, hmm?
            arrowheadSize = config.arrowheadSize / arrowLength # Because in cv it's relative size :|
            arrowWidth = int(np.abs(synapse.weight) + 1)
            arrowColor = getColorForValue(synapse.weight, config.negativeColor, config.neutralColor, config.positiveColor)
            cv.arrowedLine(canvas, (startpointX, startpointY), (endpointX, endpointY), [int(c * 255) for c in arrowColor], arrowWidth, line_type=cv.LINE_AA, tipLength=arrowheadSize)
    return imgUint8ToFloat32(canvas)

def createVisualization(canvasHeight, canvasWidth, organism, solver, observation, action, config):
    # FIXME: observation and action can be taken from organism.memory, but it's 1D only.. Hmm.
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

    obsCoords       = [(y + obsVizOffsetY,      x + obsVizOffsetX)      for y, x in obsCoords]      # Confirmed correct :)
    outputsCoords   = [(y + outputVizOffsetY,   x + outputVizOffsetX)   for y, x in outputsCoords]  # Confirmed correct :
    
    canvas = visualizeSynapses(canvas, organism, solver, obsCoords, outputsCoords, cellSize, config)

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

def getColorForValue(value, negativeColor, neutralColor, positiveColor):
    targetColor = positiveColor if value > 0 else negativeColor
    return [n + (t - n) * np.abs(np.clip(value, -1, 1)) for n, t in zip(neutralColor, targetColor)]