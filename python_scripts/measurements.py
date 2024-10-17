import subprocess
import os
from pathlib import Path


APP_NAME = "collatz"
MODE = "C 13 14 15" # possible options: "L", "D", "C {des_nr_1} {des_nr_2} ... {des_nr_n}" (e.g. "C 1 7 13")
NR_OF_STREAMS = 1
POWER_CAP = 1000
ARGUMENTS = f"A {NR_OF_STREAMS} {POWER_CAP}"
COMMAND = f"./run_scripts/run-app {APP_NAME} {MODE} {ARGUMENTS}"


if __name__ == "__main__":
    os.chdir(Path.home() / Path("parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib"))
    print(COMMAND.split(" "))
    try:
        result = subprocess.run(COMMAND.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
        print("Return Code:", e.returncode)
        print("Error Output:\n", e.stderr)
