import torch
from algorithms.algorithm import Algorithm
from algorithms.encodings import NUM_CHANNELS
from algorithms.puzzle_mappings import Puzzle_Mapping, MAX_GENERAL_COLORS
import torch.nn as nn 
    
class One_Shot_Convolutional_Model(nn.Module):
    def __init__(self, kernel_size : int, num_channels_in : int, num_channels_out : int):
        super().__init__()
        self.conv = nn.Conv2d(num_channels_in, num_channels_out, kernel_size, padding=kernel_size // 2)
        
    def forward(self, x : torch.Tensor):
        x = x.permute(0,3,1,2)
        x = self.conv(x)
        return x

class One_Shot_Convolutional_Algorithm(Algorithm):
    def __init__(self, kernel_size : int, puzzle_mapping : Puzzle_Mapping, device : torch.device, id : str):
        if puzzle_mapping == Puzzle_Mapping.GENERAL:
            num_channels_in = MAX_GENERAL_COLORS
            num_channels_out = MAX_GENERAL_COLORS
        elif puzzle_mapping == Puzzle_Mapping.UNMAPPED:
            num_channels_in = NUM_CHANNELS
            num_channels_out = NUM_CHANNELS
        network : nn.Module = One_Shot_Convolutional_Model(kernel_size, num_channels_in, num_channels_out)
        Algorithm.__init__(self, puzzle_mapping, network, 0.02, 100, device, id)