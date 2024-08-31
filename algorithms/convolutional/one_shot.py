import torch
from algorithms.algorithm import Algorithm
from algorithms.encodings import NUM_CHANNELS
import torch.nn as nn 
    
class One_Shot_Convolutional_Model(nn.Module):
    def __init__(self, kernel_size : int):
        super().__init__()
        self.conv = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, kernel_size, padding=kernel_size // 2)
        
    def forward(self, x : torch.Tensor):
        x = x.permute(0,3,1,2)
        x = self.conv(x)
        return x

class One_Shot_Convolutional_Algorithm(Algorithm):
    def __init__(self, kernel_size : int, device : torch.device):
        network : nn.Module = One_Shot_Convolutional_Model(kernel_size)
        Algorithm.__init__(self, 'general', network, 0.04, 100, device)