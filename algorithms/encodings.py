import torch
from typing import List, Tuple
NUM_CHANNELS = 10

class Arc_Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs : List[torch.Tensor], outputs : List[torch.Tensor]):
        self.inputs : List[torch.Tensor] = []
        for input in inputs:
            self.inputs.append(One_Hot_Encode(input))
        self.outputs : List[torch.Tensor] = outputs
        if len(self.inputs) != len(self.outputs):
            raise Exception("Length of input and output tensors to Arc_Dataset are not equal")
        
    def __len__(self) -> int :
        return len(self.inputs)
    
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, torch.Tensor] :
        return self.inputs[idx], self.outputs[idx]

def One_Hot_Encode(input : torch.Tensor) -> torch.Tensor :
    width : int = input.size(0)
    height : int = input.size(1)
    output : torch.Tensor = torch.zeros((width, height, NUM_CHANNELS))
    for x in range(width):
        for y in range(height):
            one_hot = torch.zeros(NUM_CHANNELS)
            val = input[x,y].item()
            one_hot[val] = 1
            output[x,y,:] = one_hot
            
    return output

def One_Hot_Decode(input : torch.tensor) -> torch.Tensor :
    width : int = input.size(0)
    height : int = input.size(1)
    output : torch.Tensor = torch.zeros((width, height), dtype=int)
    for x in range(width):
        for y in range(height):
            highest_val = -torch.inf
            highest_idx = -1
            outputs : torch.tensor = input[x,y,:]
            for z in range(NUM_CHANNELS):
                test_val = outputs[z].item()
                if test_val > highest_val:
                    highest_val = test_val
                    highest_idx = z
                    
            output[x,y] = highest_idx
            
    return output