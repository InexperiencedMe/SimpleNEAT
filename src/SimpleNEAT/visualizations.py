import numpy as np
import cv2 as cv

def createVisualization(observation, canvasStartPoint, canvasEndPoint, paddingColor=(40, 40, 40, 255)):
    # TODO: Put this in config
    coloredObservation = True
    positiveColor = (0, 255, 255, 255)
    negativeColor = (255, 0, 0, 255)
    padding = 5

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
    observationRGBA = (observationRGBA01*255).astype(np.int8)

    # 3. Calculate Height constraints
    canvasHeightRequested = abs(canvasEndPoint[1] - canvasStartPoint[1])

    # 4. Calculate cellSize (Integer division leaves a remainder)observationRGBA
    cellSize = (canvasHeightRequested - (padding * (rows + 1))) // rows
    if cellSize <= 0: cellSize = 1

    # 5. NEW: Calculate the EXACT dimensions needed for the block
    # This prevents the "stripe" at the bottom/right from integer remainders
    actualHeight = (rows * cellSize) + ((rows + 1) * padding)
    actualWidth  = (cols * cellSize) + ((cols + 1) * padding)
    
    # 6. Create the sub-canvas at the EXACT size
    obs_block = np.full((actualHeight, actualWidth, 4), paddingColor, dtype=np.uint8)

    # 7. Fill the block
    for r in range(rows):
        y_top = padding + r * (cellSize + padding)
        for c in range(cols):
            x_left = padding + c * (cellSize + padding)
            # Paste the colored square
            obs_block[y_top : y_top + cellSize, 
                      x_left : x_left + cellSize] = observationRGBA[r, c]

    return obs_block

def embedForegroundOnFrame(foreground, frame, position, globalAlpha=1.0):
    """
    Blends an RGBA foreground image onto an RGB frame using per-pixel alpha.
    """
    offsetX, offsetY = position
    fh, fw = foreground.shape[:2]

    # Ensure we don't go out of bounds of the frame
    if offsetY + fh > frame.shape[0] or offsetX + fw > frame.shape[1]:
        # Optional: Trim foreground if it exceeds frame boundaries
        fh = min(fh, frame.shape[0] - offsetY)
        fw = min(fw, frame.shape[1] - offsetX)
        foreground = foreground[:fh, :fw]

    # 1. Get the background slice
    background = frame[offsetY:offsetY + fh, offsetX:offsetX + fw]

    # 2. Extract Alpha channel and combine with globalAlpha
    # Foreground is (H, W, 4), Channel 3 is Alpha
    pixelAlpha = (foreground[:, :, 3] / 255.0) * globalAlpha
    pixelAlpha = pixelAlpha[..., np.newaxis] # Expand to (H, W, 1) for broadcasting

    # 3. Blending math: result = bg * (1-a) + fg * a
    # Convert to float for calculation, then back to uint8
    blended = (background.astype(float) * (1.0 - pixelAlpha) + 
               foreground[:, :, :3].astype(float) * pixelAlpha).astype(np.uint8)

    # 4. Paste back
    frame[offsetY:offsetY + fh, offsetX:offsetX + fw] = blended
    return frame

def percentCoordsToIdx(point: tuple[float, float], canvas: np.ndarray) -> tuple[float, float]:
    x, y = int(point[0]*canvas.shape[1]), int(point[1]*canvas.shape[0])
    return (x, y)
