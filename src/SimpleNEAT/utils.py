import os
import yaml
import attridict
import numpy as np

def ensurePath(*pathElements):
    path = os.path.join(*pathElements)
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
    return path

def loadConfig(filename, folder="configs"):
    if not filename.endswith(".yml"):
        filename += ".yml"
    
    configPath = os.path.join(os.getcwd(), folder, filename)
    
    if not os.path.exists(configPath):
        raise FileNotFoundError(f"Config '{filename}' not found in folder: {configPath}")
    
    with open(configPath, 'r') as configFile:
        config = yaml.load(configFile, Loader=yaml.FullLoader)
    
    return attridict(config)

def embedForegroundOnFrame(foreground, frame, posPercentY, posPercentX, alpha):
    offsetY, offsetX = int(posPercentY*frame.shape[0]), int(posPercentX*frame.shape[1])

    height, width, _ = foreground.shape
    background = frame[offsetY:offsetY + height, offsetX:offsetX + width]
    blended = (background * (1 - alpha) + foreground * alpha).astype(np.uint8)
    frame[offsetY:offsetY + height, offsetX:offsetX + width] = blended
    return frame




