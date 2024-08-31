from tqdm import tqdm
import json
import torch
from typing import List, Tuple, Dict, Optional
from algorithms.algorithm import Algorithm
from algorithms.convolutional.one_shot import One_Shot_Convolutional_Algorithm

def Check_Solution(file, solution):
    with open(file) as solution_file:
        solution = json.loads(solution_file)
                
def Test_Dispatcher(train_inputs : List[torch.tensor], 
                    train_outputs : List[torch.tensor], 
                    test_inputs : List[torch.tensor],
                    solutions : torch.tensor,
                    algorithms : List[Algorithm]) -> bool :     
    success_found = False
    failure_found = False
    for algorithm in algorithms:
        test_solutions = algorithm.Solve_Puzzle(train_inputs, train_outputs, test_inputs)
        if test_solutions is not None and len(test_solutions) == len(solutions):
            algo_failed_test = False
            algo_successful_test = False
            for i in range(solutions.size(0)):
                solution = solutions[i]
                test_solution = test_solutions[i]
                if not torch.equal(solution, test_solution):
                    algo_failed_test = True
                else:
                    algo_successful_test = True
            if algo_failed_test:
                failure_found = True
            elif algo_successful_test:
                success_found = True
                
                
    return success_found and not failure_found

torch.set_printoptions(linewidth=1000, edgeitems=20)
a = One_Shot_Convolutional_Algorithm(5, torch.device("cpu"))

with open("data/arc-agi_training_challenges.json") as easy_challenges:
    with open("data/arc-agi_training_solutions.json") as easy_solutions:
        challenges_json = json.load(easy_challenges)
        solutions_json = json.load(easy_solutions)
        
        successful_puzzles = 0
        failed_puzzles = 0
        for problem in tqdm(challenges_json):
            # if problem != '0962bcdd':
            #     continue
            try:
                train_inputs = []
                train_outputs = []
                test_inputs = []
                
                for t in challenges_json[problem]["train"]:
                    train_inputs.append(torch.tensor(t["input"]))
                    train_outputs.append(torch.tensor(t["output"]))
                for t in challenges_json[problem]["test"]:
                    test_inputs.append(torch.tensor(t["input"]))
                    
                solutions = torch.tensor(solutions_json[problem])
                solution_found = Test_Dispatcher(train_inputs, train_outputs, test_inputs, solutions, [a])
                if solution_found:
                    successful_puzzles += 1
                else:
                    failed_puzzles += 1
            except Exception as e:
                pass
                #print(e)
                
        print(f"Successfully completed {successful_puzzles} out of {successful_puzzles + failed_puzzles} easy puzzles")
        print(f"Success rate of {successful_puzzles / (successful_puzzles + failed_puzzles) * 100:.1f}%")
    