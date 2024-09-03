import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from algorithms.puzzle_mappings import Map_General_Mapping, Unmap_General_Mapping, Apply_Map, Puzzle_Mapping, MAX_GENERAL_COLORS
from algorithms.encodings import Arc_Dataset, NUM_CHANNELS, One_Hot_Encode, One_Hot_Decode

def Puzzle_Loss(predictions : torch.Tensor, actual : torch.Tensor) -> torch.Tensor:
    predictions = predictions.permute(0,2,3,1)
    # Reshape into form reqiuired by CrossEntropyLoss
    puzzle_width : int = predictions.size(1)
    puzzle_height : int = predictions.size(2)
    predictions = predictions.reshape((puzzle_width * puzzle_height, predictions.size(3)))
    actual = actual.reshape((puzzle_width * puzzle_height,))
    # Calculate Loss
    cel = nn.CrossEntropyLoss(reduction='none')
    loss : torch.Tensor = cel(predictions, actual)
    return loss.mean()

class Algorithm:
    def __init__(self,
                 map_type : Puzzle_Mapping,
                 network : nn.Module,
                 lr : float,
                 num_epochs : int,
                 device : torch.device,
                 id : str):
        self.map_type = map_type
        self.network = network
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.id = id
    
    # This function is meant to be overridden
    def Solve_Puzzle(self,
                     train_inputs : List[torch.tensor], 
                    train_outputs : List[torch.tensor], 
                    test_inputs : List[torch.tensor],
                    print_training : bool = False) -> Tuple[Optional[List[torch.Tensor]],
                                                        Optional[str]]:
        
        # Map Data
        train_inputs_variants = []
        train_outputs_variants = []
        if self.map_type == Puzzle_Mapping.GENERAL:
            num_colors = MAX_GENERAL_COLORS
            for n in range(len(train_inputs)):
                general_mapping = Map_General_Mapping(train_inputs[n], True, False)
                for mapped_input, map in general_mapping:
                    train_inputs_variants.append(mapped_input)
                    train_outputs_variants.append(Apply_Map(train_outputs[n], map))
        elif self.map_type == Puzzle_Mapping.GENERAL_NO_BACKGROUND:
            return None, "General No Background Not Implemented"
            for n in range(len(train_inputs)):
                mapped_input, map = Map_General_Mapping(train_inputs[n], False)
                train_inputs[n] = mapped_input
                train_outputs[n] = Apply_Map(train_outputs[n], map)
        elif self.map_type == Puzzle_Mapping.UNMAPPED:
            num_colors = NUM_CHANNELS
            for n in range(len(train_inputs)):
                train_inputs_variants.append(train_inputs[n])
                train_outputs_variants.append(train_outputs[n])
        else:
            return None, "Invalid Algorithm"
        
        # Load Training Data
        validation_input : torch.Tensor = train_inputs_variants[-1]
        train_inputs_variants = train_inputs_variants[:-1]
        validation_output : torch.Tensor = train_outputs_variants[-1]
        train_outputs_variants = train_outputs_variants[:-1]
        
        train_dataset : Arc_Dataset = Arc_Dataset(train_inputs_variants, train_outputs_variants, num_colors)
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
        self.network = self.network.to(self.device)
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
        # one_hot_validate : torch.Tensor = One_Hot_Encode(train_inputs[0], num_colors).to(self.device)
        # model_output : torch.Tensor = self.network(one_hot_validate.unsqueeze(0))
        # model_output = model_output.squeeze(0)
        # model_output = model_output.permute(1,2,0)
        # model_decoded : torch.Tensor = One_Hot_Decode(model_output, num_colors)
        # print(model_decoded)
        
        # Validate Model
        self.network.eval()
        one_hot_validate : torch.Tensor = One_Hot_Encode(validation_input, num_colors).to(self.device)
        model_output : torch.Tensor = self.network(one_hot_validate.unsqueeze(0))
        model_output = model_output.squeeze(0)
        model_output = model_output.permute(1,2,0)
        model_decoded : torch.Tensor = One_Hot_Decode(model_output, num_colors)
        if not torch.equal(model_decoded, validation_output):
            # print(model_decoded)
            # print(validation_output)
            # print("Solution Mismatch")
            return None, "Validation_Step" # If validation gives wrong answer, likely don't have right algorithm
        
        # Solve Test problems, given that validation was correct
        solutions = []
        for n in range(len(test_inputs)):
            if self.map_type == Puzzle_Mapping.GENERAL:
                mapped_input, map = Map_General_Mapping(test_inputs[n], True, True)[0]
            elif self.map_type == Puzzle_Mapping.GENERAL_NO_BACKGROUND:
                mapped_input, map = Map_General_Mapping(test_inputs[n], False)
            elif self.map_type == Puzzle_Mapping.UNMAPPED:
                mapped_input = test_inputs[n]
                map = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9}
            
            one_hot_validate : torch.Tensor = One_Hot_Encode(mapped_input, num_colors).to(self.device)
            model_output : torch.Tensor = self.network(one_hot_validate.unsqueeze(0))
            model_output = model_output.squeeze(0)
            model_output = model_output.permute(1,2,0)
            model_decoded : torch.Tensor = One_Hot_Decode(model_output, num_colors)
            solutions.append(Unmap_General_Mapping(model_decoded, map))
        
        return solutions, None