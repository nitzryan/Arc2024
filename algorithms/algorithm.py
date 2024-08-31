import torch
import torch.nn as nn
from typing import List, Optional
from algorithms.puzzle_mappings import Map_General_Mapping, Unmap_General_Mapping, Apply_Map
from algorithms.encodings import Arc_Dataset, NUM_CHANNELS, One_Hot_Encode, One_Hot_Decode

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

class Algorithm:
    def __init__(self,
                 map_type : str,
                 network : nn.Module,
                 lr : float,
                 num_epochs : int,
                 device : torch.device):
        self.map_type = map_type
        self.network = network
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        pass
    
    # This function is meant to be overridden
    def Solve_Puzzle(self,
                     train_inputs : List[torch.tensor], 
                    train_outputs : List[torch.tensor], 
                    test_inputs : List[torch.tensor],
                    print_training : bool = False) -> Optional[List[torch.Tensor]] :
        
        # Map Data
        if self.map_type == 'general':
            for n in range(len(train_inputs)):
                mapped_input, map = Map_General_Mapping(train_inputs[n], True)
                train_inputs[n] = mapped_input
                train_outputs[n] = Apply_Map(train_outputs[n], map)
        elif self.map_type == 'general_no_background':
            for n in range(len(train_inputs)):
                mapped_input, map = Map_General_Mapping(train_inputs[n], False)
                train_inputs[n] = mapped_input
                train_outputs[n] = Apply_Map(train_outputs[n], map)
        else:
            return None
        
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
        with torch.no_grad():
            for p in self.network.parameters():
                p.zero_()
        self.network.train()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
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
            if print_training:
                print(f"Epoch {epoch:3d} Loss={total_loss:.3f}")
        
        # Sanity check for debugging
        # self.network.eval()
        # one_hot_validate : torch.Tensor = One_Hot_Encode(train_inputs[0])
        # model_output : torch.Tensor = self.network(one_hot_validate.unsqueeze(0))
        # model_output = model_output.squeeze(0)
        # model_output = model_output.permute(1,2,0)
        # model_decoded : torch.Tensor = One_Hot_Decode(model_output)
        
        # Validate Model
        self.network.eval()
        one_hot_validate : torch.Tensor = One_Hot_Encode(validation_input)
        model_output : torch.Tensor = self.network(one_hot_validate.unsqueeze(0))
        model_output = model_output.squeeze(0)
        model_output = model_output.permute(1,2,0)
        model_decoded : torch.Tensor = One_Hot_Decode(model_output)
        if not torch.equal(model_decoded, validation_output):
            # print(model_decoded)
            # print(validation_output)
            # print("Solution Mismatch")
            return None # If validation gives wrong answer, likely don't have right algorithm
        
        # Solve Test problems, given that validation was correct
        solutions = []
        for n in range(len(test_inputs)):
            mapped_input, map = Map_General_Mapping(test_inputs[n], True)
            
            one_hot_validate : torch.Tensor = One_Hot_Encode(mapped_input)
            model_output : torch.Tensor = self.network(one_hot_validate.unsqueeze(0))
            model_output = model_output.squeeze(0)
            model_output = model_output.permute(1,2,0)
            model_decoded : torch.Tensor = One_Hot_Decode(model_output)
            solutions.append(Unmap_General_Mapping(model_decoded, map))
        
        return solutions