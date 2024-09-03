import torch
from typing import Tuple, Dict, List
from enum import Enum
from itertools import permutations

MAX_GENERAL_COLORS = 5 # Includes background

class Puzzle_Mapping(Enum):
    GENERAL = 1
    GENERAL_NO_BACKGROUND = 2
    UNMAPPED = 3

def Map_General_Mapping(puzzle : torch.Tensor, 
                        hardcode_background : bool,
                        only_first : bool) -> List[Tuple[torch.Tensor, Dict]] :
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
            
    # These mappings are arbitrary, so we can create pairs from every possible ordering
    if not hardcode_background:
        raise Exception("Function Not Implemented for hardcoding background")
    start_idx = 1
    num_colors = nextMappedIdx - start_idx
    if num_colors >= MAX_GENERAL_COLORS:
        raise Exception(f"Too Many Colors for Map_General_Mapping : {num_colors}")
    idx_orderings = permutations(range(start_idx, nextMappedIdx), num_colors)
    
    puzzle_mappings = []
    puzzle_mappings.append((mapped_puzzle, mapping))
    if only_first:
        return puzzle_mappings
    
    for ordering in idx_orderings:
        this_map = {0:0}
        for keys, values in mapping.items():
            if keys == 0:
                continue
            this_map[keys] = ordering[values - 1]
        this_puzzle = Apply_Map(puzzle, this_map)
        puzzle_mappings.append((this_puzzle, this_map))
    return puzzle_mappings

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
    width : int = puzzle.size(0)
    height : int = puzzle.size(1)
    puzzle = puzzle.reshape((width * height))
    mapped_puzzle = puzzle.clone()
    
    for key, value in map.items():
        mapped_puzzle[puzzle == key] = value
        
    mapped_puzzle = mapped_puzzle.reshape((width, height))
    #print(mapped_puzzle.shape)
    return mapped_puzzle
    
    for x in range(puzzle.size(0)):
        for y in range(puzzle.size(1)):
            val = puzzle[x,y].item()
            mapped_puzzle[x,y] = map[val]
    return mapped_puzzle