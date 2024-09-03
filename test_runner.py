from tqdm import tqdm
import json
import torch
from algorithms.convolutional.one_shot import One_Shot_Convolutional_Algorithm
from algorithms.puzzle_mappings import Puzzle_Mapping
from database.database import Generate_DB, Open_Db_Browser
from test_dispatcher import Test_Dispatcher

torch.set_printoptions(linewidth=1000, edgeitems=20)
db_filename = "database/test_results.db"
Generate_DB(db_filename)
device = torch.device("cpu")
a = One_Shot_Convolutional_Algorithm(5, Puzzle_Mapping.GENERAL, device, "One_Shot_Convolutional_Mapped_5")
b = One_Shot_Convolutional_Algorithm(3, Puzzle_Mapping.GENERAL, device, "One_Shot_Convolutional_Mapped_3")
c = One_Shot_Convolutional_Algorithm(7, Puzzle_Mapping.GENERAL, device, "One_Shot_Convolutional_Mapped_7")
d = One_Shot_Convolutional_Algorithm(9, Puzzle_Mapping.GENERAL, device, "One_Shot_Convolutional_Mapped_9")
e = One_Shot_Convolutional_Algorithm(5, Puzzle_Mapping.UNMAPPED, device, "One_Shot_Convolutional_Unmapped_5")
f = One_Shot_Convolutional_Algorithm(3, Puzzle_Mapping.UNMAPPED, device, "One_Shot_Convolutional_Unmapped_3")
g = One_Shot_Convolutional_Algorithm(7, Puzzle_Mapping.UNMAPPED, device, "One_Shot_Convolutional_Unmapped_7")
h = One_Shot_Convolutional_Algorithm(9, Puzzle_Mapping.UNMAPPED, device, "One_Shot_Convolutional_Unmapped_9")

algorithms = [a,b,c,d,e,f,g,h]
algorithms = [a,e]
with open("data/arc-agi_training_challenges.json") as easy_challenges:
    with open("data/arc-agi_training_solutions.json") as easy_solutions:
        challenges_json = json.load(easy_challenges)
        solutions_json = json.load(easy_solutions)
        
        successful_puzzles = 0
        failed_puzzles = 0
        for problem in tqdm(challenges_json):
            # if problem != '05269061':
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
                solution_found = Test_Dispatcher(train_inputs, train_outputs, test_inputs, solutions, algorithms, problem)
                if solution_found:
                    successful_puzzles += 1
                else:
                    failed_puzzles += 1
            except Exception as e:
                pass
                #print(e)
                
        print(f"Successfully completed {successful_puzzles} out of {successful_puzzles + failed_puzzles} easy puzzles")
        print(f"Success rate of {successful_puzzles / (successful_puzzles + failed_puzzles) * 100:.1f}%")
    
Open_Db_Browser(db_filename)