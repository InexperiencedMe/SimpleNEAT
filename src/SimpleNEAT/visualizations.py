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
        observationRGBA = np.concatenate((observationRGB, np.ones_like(observationRGB[:, :, 0:1])), axis=-1)
    else:
        observation = ((values + 1) / 2.0)
        observationRGBA = np.stack([observation, observation, observation, np.ones_like(observation)], axis=-1)

    observationVisualizationHeight = (rows * cellSize) + ((rows + 1) * config.gridThickness)
    observationVisualizationWidth  = (cols * cellSize) + ((cols + 1) * config.gridThickness)
    
    observationVisualization = np.full((observationVisualizationHeight, observationVisualizationWidth, 4), config.gridColor, dtype=np.float32)

    grid = []
    offset = cellSize // 2 # To make grid from cell centers
    for row in range(rows):
        y_topLeft = config.gridThickness + row * (cellSize + config.gridThickness)
        for column in range(cols):
            x_topLeft = config.gridThickness + column * (cellSize + config.gridThickness)
            observationVisualization[y_topLeft:y_topLeft + cellSize, x_topLeft:x_topLeft + cellSize] = observationRGBA[row, column]
            grid.append((y_topLeft + offset, x_topLeft + offset))

    return observationVisualization, grid

def translateNeuronToCoords(neuron, organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize, gridThickness, alreadyCalculatedPositions):
    if neuron in alreadyCalculatedPositions: return alreadyCalculatedPositions[neuron]

    if neuron < organism.inputSize:
        inputY, inputX = obsCoords[neuron]
        return inputY, inputX
    
    if neuron == organism.biasNeuron:
        lastObsCellY, lastObsCellX = obsCoords[-1]
        return lastObsCellY, lastObsCellX + cellSize*4
    
    if neuron < organism.inputSizeWithBias + organism.outputSize:
        outputY, outputX = outputsCoords[neuron - organism.inputSizeWithBias]
        return outputY, outputX
    
    source, destination = neuronToLinkMap[neuron]
    sourceY, sourceX            = translateNeuronToCoords(source,       organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize, gridThickness, alreadyCalculatedPositions)
    destinationY, destinationX  = translateNeuronToCoords(destination,  organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize, gridThickness, alreadyCalculatedPositions)

    return (sourceY + destinationY) // 2, (sourceX + destinationX) // 2

def calculateNeuronPositions(organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize, gridThickness):
    neuronPositions = {}
    for neuron in organism.neurons:
        neuronPositions[neuron] = translateNeuronToCoords(neuron, organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize, gridThickness, neuronPositions)
    return neuronPositions

def visualizeSynapses(canvas, organism, neuronPositions, config):
    synapseSignalPairs = ((synapse, organism.lastSignals[id]) for id, synapse in organism.synapses.items() if synapse.enabled)
    for synapse, signal in sorted(synapseSignalPairs, key=lambda pair: abs(pair[1])):
        startpointY, startpointX    = neuronPositions[synapse.source]
        endpointY, endpointX        = neuronPositions[synapse.destination]

        arrowLength = np.sqrt((startpointY - endpointY)**2 + (startpointX - endpointX)**2)
        if arrowLength != 0: # TODO: We could potentially display self recursion, hmm?
            arrowWidth = int(np.abs(synapse.weight*3) + 2)
            arrowColor = getColorForValue(signal, config.negativeColor, config.neutralSynapseColor, config.positiveColor)
            cv.line(canvas, (startpointX, startpointY), (endpointX, endpointY), arrowColor, arrowWidth, lineType=cv.LINE_AA)
    return canvas

def drawHiddenNeurons(canvas, organism, neuronPositions, cellSize, config):
    for neuron in [organism.biasNeuron] + [n for n in organism.neurons if n > organism.inputSizeWithBias + organism.outputSize]:
        neuronY, neuronX = neuronPositions[neuron]
        y_topLeft, x_topLeft = neuronY - (cellSize//2), neuronX - (cellSize//2)
        canvas[y_topLeft - config.gridThickness:y_topLeft + cellSize + config.gridThickness, x_topLeft - config.gridThickness:x_topLeft + cellSize + config.gridThickness] = config.gridColor
        canvas[y_topLeft:y_topLeft + cellSize, x_topLeft:x_topLeft + cellSize] = getColorForValue(organism.memory[neuron], config.negativeColor, config.neutralNeuronColor, config.positiveColor)
    return canvas

def createVisualization(canvasHeight, canvasWidth, organism, solver, observation, action, config):
    # TODO: To vastly optimize this and calculate neuron positions only once, I need internal state. Visualizer has to be class
    # TODO: observation and action can be taken from organism.memory, but it's 1D only.. Hmm.
    canvas = np.zeros((canvasHeight, canvasWidth, 4), dtype=np.float32)

    cleanObservation, obsRows, obsCols  = preprocessValuesForGridVisualization(observation)
    cleanAction, actionRows, actionCols = preprocessValuesForGridVisualization(action)
    cellSize = (canvasHeight - config.gridThickness * (obsRows + 1)) // obsRows

    obsViz, obsCoords = createVisualizationGrid(cleanObservation, obsRows, obsCols, cellSize, config)
    obsVizHeight, obsVizWidth = obsViz.shape[:2]
    obsVizOffsetX = 0
    obsVizOffsetY = (canvasHeight - obsVizHeight) // 2

    outputViz, outputsCoords = createVisualizationGrid(cleanAction, actionRows, actionCols, cellSize, config)
    outputVizHeight, outputVizWidth = outputViz.shape[:2]
    outputVizOffsetX = canvasWidth - outputVizWidth
    outputVizOffsetY = (canvasHeight - outputVizHeight) // 2

    obsCoords       = [(y + obsVizOffsetY,      x + obsVizOffsetX)      for y, x in obsCoords]
    outputsCoords   = [(y + outputVizOffsetY,   x + outputVizOffsetX)   for y, x in outputsCoords]
    
    neuronToLinkMap = {v: k for k, v in solver.tracker.novelNeurons.items()}
    neuronPositions = calculateNeuronPositions(organism, neuronToLinkMap, obsCoords, outputsCoords, cellSize, config.gridThickness) # Need to save this output somehow instead of recalculating each frame
    canvas = visualizeSynapses(canvas, organism, neuronPositions, config)

    canvas = drawHiddenNeurons(canvas, organism, neuronPositions, cellSize//3, config)
    canvas[obsVizOffsetY:obsVizOffsetY + obsVizHeight, obsVizOffsetX:obsVizOffsetX + obsVizWidth] = obsViz
    canvas[outputVizOffsetY:outputVizOffsetY + outputVizHeight, outputVizOffsetX:outputVizOffsetX + outputVizWidth] = outputViz

    return canvas

def embedForegroundOnFrame(foreground, frame, position, globalAlpha=1.0):
    offsetX, offsetY = position
    foregroundHeight, foregroundWidth = foreground.shape[:2]

    backgroundToBeBlended = frame[offsetY:offsetY + foregroundHeight, offsetX:offsetX + foregroundWidth]
    pixelAlpha = foreground[:, :, 3:4] * globalAlpha

    blended = backgroundToBeBlended * (1.0 - pixelAlpha) + foreground[:, :, :3] * pixelAlpha
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
    return tuple(float(n + (t - n) * np.abs(np.clip(value, -1, 1))) for n, t in zip(neutralColor, targetColor))