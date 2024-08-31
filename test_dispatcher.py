from database.database import Log_Test_Result, Log_Puzzle_Result
from typing import List
import torch
from algorithms.algorithm import Algorithm

def Test_Dispatcher(train_inputs : List[torch.Tensor], 
                    train_outputs : List[torch.Tensor], 
                    test_inputs : List[torch.Tensor],
                    solutions : torch.Tensor,
                    algorithms : List[Algorithm],
                    puzzle_id : str) -> bool :     
    success_found = False
    failure_found = False
    for algorithm in algorithms:
        try:
            test_solutions, test_info = algorithm.Solve_Puzzle(train_inputs, train_outputs, test_inputs)
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
                    Log_Test_Result(puzzle_id, algorithm.id, False, True, False, False, None, None)
                elif algo_successful_test:
                    success_found = True
                    Log_Test_Result(puzzle_id, algorithm.id, True, False, False, False, None, None)
            else: # No results, log why
                if test_info == "Validation_Step":
                    Log_Test_Result(puzzle_id, algorithm.id, False, False, False, True, None, None)
                elif test_info == "Assumption_Step":
                    Log_Test_Result(puzzle_id, algorithm.id, False, False, True, False, None, None)
                else:
                    Log_Test_Result(puzzle_id, algorithm.id, False, False, False, False, None, test_info)
        except Exception as e:
            Log_Test_Result(puzzle_id, algorithm.id, False, False, False, False, e.__str__(), None)
    
    Log_Puzzle_Result(puzzle_id, success_found, failure_found)            
                
    return success_found and not failure_found