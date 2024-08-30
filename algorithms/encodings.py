import torch
NUM_CHANNELS = 10

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