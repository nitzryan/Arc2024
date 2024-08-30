import torch
from algorithms.algorithm import Algorithm
from algorithms.puzzle_mappings import Map_General_Mapping, Unmap_General_Mapping, Apply_Map
from algorithms.encodings import One_Hot_Encode, One_Hot_Decode
from typing import List, Optional, Tuple
import torch.utils.data.dataset
import torch.nn as nn

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
    

    
class One_Shot_Convolutional_Model(nn.Module):
    def __init__(self, kernel_size : int):
        super().__init__()
        self.conv = nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, kernel_size, padding=kernel_size // 2)
        
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.conv(x)
        return x

def Puzzle_Loss(predictions : torch.Tensor, actual : torch.Tensor) -> torch.Tensor:
    predictions = predictions.permute(0,2,3,1)
    # Reshape into form reqiuired by CrossEntropyLoss
    puzzle_width : int = predictions.size(1)
    puzzle_height : int = predictions.size(2)
    predictions = predictions.reshape((puzzle_width * puzzle_height, NUM_CHANNELS))
    actual = actual.reshape((puzzle_width * puzzle_height,))
    # Calculate Loss
    cel = nn.CrossEntropyLoss(reduction='none')
    loss : torch.Tensor = cel(predictions, actual)
    return loss.mean()

class One_Shot_Convolutional_Algorithm(Algorithm):
    def __init__(self, kernel_size : int, device : torch.device):
        Algorithm.__init__(self)
        self.device : torch.device = device
        self.network : nn.Module = One_Shot_Convolutional_Model(kernel_size)
        
    def Solve_Puzzle(self,
                     train_inputs : List[torch.Tensor], 
                    train_outputs : List[torch.Tensor], 
                    test_inputs : List[torch.Tensor]) -> Optional[torch.Tensor] :
        
        # Map Training Data
        for n in range(len(train_inputs)):
            mapped_input, map = Map_General_Mapping(train_inputs[n], True)
            train_inputs[n] = mapped_input
            train_outputs[n] = Apply_Map(train_outputs[n], map)
        
        # Load Training Data
        validation_input : torch.Tensor = train_inputs[-1]
        train_inputs = train_inputs[:-1]
        validation_output : torch.Tensor = train_outputs[-1]
        train_outputs = train_outputs[:-1]
        
        train_dataset : Arc_Dataset = Arc_Dataset(train_inputs, train_outputs)
        train_generator : torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1, # Prevent issues with different shaped puzzles running concurrently
            shuffle=True
        )
        # Train Model
        self.network.to(self.device)
        self.network.train()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.02)
        for epoch in range(25):
            total_loss : float = 0
            num_batches : int = 0
            for input_data, output_data in train_generator:
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)
                prediction = self.network(input_data)
                optimizer.zero_grad()
                loss : torch.Tensor = Puzzle_Loss(prediction, output_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            print(f"Epoch {epoch:3d} Loss={total_loss:.3f}")
        
        # Sanity Check
        self.network.eval()
        one_hot_validate : torch.Tensor = One_Hot_Encode(train_inputs[0])
        model_output : torch.Tensor = self.network(one_hot_validate.unsqueeze(0))
        model_decoded : torch.Tensor = One_Hot_Decode(model_output.squeeze(0))
        # print(model_decoded)
        # print(train_outputs[0])
        
        # Validate Model
        self.network.eval()
        one_hot_validate : torch.Tensor = One_Hot_Encode(validation_input)
        model_output : torch.Tensor = self.network(one_hot_validate.unsqueeze(0))
        model_decoded : torch.Tensor = One_Hot_Decode(model_output.squeeze(0))
        if not torch.equal(model_decoded, validation_output):
            # print(model_decoded)
            # print(validation_output)
            print("Solution Mismatch")
            return None # If validation gives wrong answer, likely don't have right algorithm
        
        # Solve Test problems
        # TODO : Need to refactor to handle multiple test cases
        one_hot_validate : torch.Tensor = One_Hot_Encode(test_inputs[0])
        model_output : torch.Tensor = self.network(one_hot_validate.unsqueeze(0))
        model_decoded : torch.Tensor = One_Hot_Decode(model_output.squeeze(1))
        
        return model_decoded
