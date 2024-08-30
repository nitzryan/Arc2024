import torch
from typing import Tuple, Dict

def Map_General_Mapping(puzzle : torch.Tensor, hardcode_background : bool) -> Tuple[torch.Tensor, Dict] :
    # Setup Map
    mapping = {}
    if hardcode_background:
        mapping[0] = 0
    nextMappedIdx = 1
    
    # Iterate through puzzle, adding to mapping
    # When new symbol found, add to mapping
    mapped_puzzle = puzzle.clone()
    for x in range(puzzle.size(0)):
        for y in range(puzzle.size(1)):
            val = puzzle[x,y].item()
            if not val in mapping.keys():
                mapping[val] = nextMappedIdx
                nextMappedIdx += 1
            mapped_puzzle[x,y] = mapping[val]
            
    return (mapped_puzzle, mapping)

def Unmap_General_Mapping(mapped_puzzle : torch.Tensor, mapping : Dict) -> torch.Tensor : 
    # Reverse the mapping dict
    unmapping = {}
    for key, value in mapping.items():
        unmapping[value] = key
        
    # Apply reversed mapping
    unmapped_puzzle = mapped_puzzle.clone()
    for x in range(mapped_puzzle.size(0)):
        for y in range(mapped_puzzle.size(1)):
            val = mapped_puzzle[x,y].item()
            unmapped_puzzle[x,y] = unmapping[val]
            
    return unmapped_puzzle

def Apply_Map(puzzle : torch.Tensor, map : Dict) -> torch.Tensor :
    mapped_puzzle = puzzle.clone()
    for x in range(puzzle.size(0)):
        for y in range(puzzle.size(1)):
            val = puzzle[x,y].item()
            mapped_puzzle[x,y] = map[val]
    return mapped_puzzle