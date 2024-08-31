import sqlite3
import os
from typing import Optional

_db : sqlite3.Connection

def Generate_DB(filename : str) -> None:
    global _db
    # Delete, create file
    if os.path.exists(filename):
        os.remove(filename)
    
    f = open(filename, "w")
    f.close()
    
    # Initialize connection and DB
    _db = sqlite3.connect(filename)
    cursor : sqlite3.Cursor = _db.cursor()
    cursor.execute('''
                   CREATE TABLE "TestResults" (
                        "PuzzleId" TEXT,
                        "TestId" TEXT,
                        "TestPassed" INTEGER,
                        "TestWrong" INTEGER,
                        "TestAssumptionFailed" INTEGER,
                        "TestValidationFailed" INTEGER,
                        "TestException" TEXT,
                        "AlgorithmException" TEXT,
                        PRIMARY KEY("PuzzleId","TestId")
                   );
                   ''')
    cursor.execute('''
                   CREATE TABLE "PuzzleResults" (
                       "PuzzleId" TEXT,
                       "AnyPassed" INTEGER,
                       "AnyWrong" INTEGER,
                       "OnlyPassed" INTEGER,
                       PRIMARY KEY("PuzzleId")
                   );
                   ''')
    _db.commit()
    
def Log_Test_Result(puzzle_id : str,
                    test_id : str,
                    test_passed : bool,
                    test_wrong : bool,
                    test_assumption : bool,
                    test_validation : bool,
                    test_exception : Optional[str],
                    algorithm_exception : Optional[str]) -> None:
    
    try:
        cursor = _db.cursor()
        cursor.execute("INSERT INTO TestResults VALUES(?,?,?,?,?,?,?,?)", (puzzle_id, test_id, test_passed, test_wrong, test_assumption, test_validation, test_exception, algorithm_exception))
        _db.commit()
    except Exception as e:
        print(e)
        
def Log_Puzzle_Result(puzzle_id : str, test_passed : bool, test_failed : bool) -> None :
    try:
        cursor = _db.cursor()
        cursor.execute("INSERT INTO PuzzleResults VALUES(?,?,?,?)", (puzzle_id, test_passed, test_failed, test_passed and not test_failed))
        _db.commit()
    except Exception as e:
        print(e)