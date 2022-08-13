import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(NeuralNet, self).__init__()
        
        # Neural network contains 4 activation layers, 1) Linear, 2) RELU, 3) Linear, 4) Sigmoid
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.RELU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

