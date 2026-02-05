class InnovationTracker:
    def __init__(self, inputSize, outputSize):
        self.novelSynapses  = {}  # {(source, destination): novelSynapseID}
        self.novelNeurons   = {}  # {(source, destination): novelNeuronID}
        self.neuronCount    = inputSize + outputSize
        self.synapseCount   = 0

    def getSynapseID(self, source, destination):
        link = (source, destination)
        if link not in self.novelSynapses:
            self.novelSynapses[link] = self.synapseCount
            self.synapseCount += 1
        return self.novelSynapses[link]
    
    def getNeuronID(self, source, destination):
        splitKey = (source, destination)
        if splitKey not in self.novelNeurons:
            self.novelNeurons[splitKey] = self.neuronCount
            self.neuronCount += 1
        return self.novelNeurons[splitKey]