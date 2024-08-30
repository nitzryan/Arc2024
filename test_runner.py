from tqdm import tqdm
import json
import torch
from typing import List, Tuple, Dict, Optional

class Algorithm:
    def __init__():
        pass
    
    def Solve_Puzzle(self,
                     train_inputs : List[torch.tensor], 
                    train_outputs : List[torch.tensor], 
                    test_inputs : List[torch.tensor]) -> Optional[torch.Tensor] :
        return None

def Check_Solution(file, solution):
    with open(file) as solution_file:
        solution = json.loads(solution_file)
                
def Test_Dispatcher(train_inputs : List[torch.tensor], 
                    train_outputs : List[torch.tensor], 
                    test_inputs : List[torch.tensor],
                    solution : torch.tensor,
                    algorithms : List[Algorithm]) -> bool :     
    success_found = False
    failure_found = False
    for algorithm in algorithms:
        test_solution = algorithm.Solve_Puzzle(train_inputs, train_outputs, test_inputs)
        if test_solution is not None:
            if torch.equal(solution, test_solution):
                success_found = True
            else:
                success_found = False
                
    return success_found and not failure_found


with open("data/arc-agi_training_challenges.json") as easy_challenges:
    with open("data/arc-agi_training_solutions.json") as easy_solutions:
        challenges_json = json.load(easy_challenges)
        solutions_json = json.load(easy_solutions)
        
        successful_puzzles = 0
        failed_puzzles = 0
        for problem in tqdm(challenges_json):
            try:
                train_inputs = []
                train_outputs = []
                test_inputs = []
                
                for t in challenges_json[problem]["train"]:
                    train_inputs.append(torch.tensor(t["input"]))
                    train_outputs.append(torch.tensor(t["output"]))
                for t in challenges_json[problem]["test"]:
                    test_inputs.append(torch.tensor(t["input"]))
                    
                solution = torch.tensor(solutions_json[problem])
                solution_found = Test_Dispatcher(train_inputs, train_outputs, test_inputs, solution, [])
                if solution_found:
                    successful_puzzles += 1
                else:
                    failed_puzzles += 1
            except:
                pass
                
        print(f"Successfully completed {successful_puzzles} out of {successful_puzzles + failed_puzzles} easy puzzles")
        print(f"Success rate of {successful_puzzles / (successful_puzzles + failed_puzzles) * 100:.1f}%")
    