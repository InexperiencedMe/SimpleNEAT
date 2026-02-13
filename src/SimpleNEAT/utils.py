import os
import yaml
import attridict

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
