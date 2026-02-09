import os
import yaml
from attridict import attridict

def loadConfig(filename, folder="configs"):
    if not filename.endswith(".yml"):
        filename += ".yml"
    
    configPath = os.path.join(os.getcwd(), folder, filename)
    
    if not os.path.exists(configPath):
        raise FileNotFoundError(f"Config '{filename}' not found in folder: {configPath}")
    
    with open(configPath, 'r') as configFile:
        config = yaml.load(configFile, Loader=yaml.FullLoader)
    
    return attridict(config)