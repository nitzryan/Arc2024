import torch
from typing import List, Tuple
NUM_CHANNELS = 10

class Arc_Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs : List[torch.Tensor], outputs : List[torch.Tensor], num_colors : int):
        self.inputs : List[torch.Tensor] = []
        for input in inputs:
            self.inputs.append(One_Hot_Encode(input, num_colors))
        self.outputs : List[torch.Tensor] = outputs
        if len(self.inputs) != len(self.outputs):
            raise Exception("Length of input and output tensors to Arc_Dataset are not equal")
        
    def __len__(self) -> int :
        return len(self.inputs)
    
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, torch.Tensor] :
        return self.inputs[idx], self.outputs[idx]

def One_Hot_Encode(input : torch.Tensor, num_colors : int) -> torch.Tensor :
    return torch.nn.functional.one_hot(input, num_classes=num_colors).float()

def One_Hot_Decode(input : torch.tensor, num_colors : int) -> torch.Tensor :
    width : int = input.size(0)
    height : int = input.size(1)
    input = input.reshape(width * height, input.size(2))
    output = torch.argmax(input, dim=1)
    output = output.reshape(width, height)
    return output
    
    output : torch.Tensor = torch.zeros((width, height), dtype=int)
    for x in range(width):
        for y in range(height):
            highest_val = -torch.inf
            highest_idx = -1
            outputs : torch.tensor = input[x,y,:]
            for z in range(num_colors):
                test_val = outputs[z].item()
                if test_val > highest_val:
                    highest_val = test_val
                    highest_idx = z
                    
            output[x,y] = highest_idx
            
    return output