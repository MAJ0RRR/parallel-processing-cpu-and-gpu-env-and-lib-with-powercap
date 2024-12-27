import subprocess
import sys
import os
import re
from pathlib import Path

ENERGY_UJ_FILE = "/sys/class/powercap/intel-rapl:0/energy_uj"
NUMBER_OF_RUNS = 1
PROBLEM_SIZE = 960_000_000
NUMBER_OF_STREAMS = 2
POWERCAP = 0
NUMBER_OF_NODES = 2

def read_energy(file_path: str = ENERGY_UJ_FILE):
    """Reads the energy value from the given file."""
    try:
        with open(file_path, 'r') as file:
            return float(file.read())
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except PermissionError:
        print(f"Error: Insufficient permissions to read file '{file_path}'.")
    except Exception as e:
        print(f"Unexpected error reading energy file: {e}")
    return None

def read_main_elapsed_time(stderr_output: str):
    match = re.search(r'Main elapsed time=([\d.]+)', stderr_output)
    if match:
        return float(match.group(1))
    print("Error: Could not find 'Main elapsed time' in stderr output.")
    return None

def run_script(command: str):
    try:
        result = subprocess.run(
            command.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Command: {e.cmd}")
        print(f"Return Code: {e.returncode}")
        print(f"Error Output:\n{e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    os.chdir(Path.home() / "projekt_badawczy/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib")
    commands = [
        f"./run_scripts/run-app collatz B {NUMBER_OF_NODES} --cpu-enabled=0 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=6000000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling=1",
        f"./run_scripts/run-app collatz B {NUMBER_OF_NODES} --cpu-enabled=1 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=6000000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling=1",
        f"./run_scripts/run-app vecadd B {NUMBER_OF_NODES} --cpu-enabled=0 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=600000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling=100",
        f"./run_scripts/run-app vecadd B {NUMBER_OF_NODES} --cpu-enabled=1 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=600000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling=100",
        f"./run_scripts/run-app vecmaxdiv B {NUMBER_OF_NODES} --cpu-enabled=0 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=6000000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling=100",
        f"./run_scripts/run-app vecmaxdiv B {NUMBER_OF_NODES} --cpu-enabled=1 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=6000000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling=100",
    ]
    
    for command in commands:
        print(f"[COMMAND] {command}")
        energy_before = read_energy()
        print(f"Energy before: {energy_before}")
        total_main_elapsed_time = 0
        for _ in range(NUMBER_OF_RUNS):
            stderr_output = run_script(command=command)
            elapsed_time = read_main_elapsed_time(stderr_output)
            if elapsed_time is None:
                print("Skipping this run due to error parsing elapsed time.")
                continue
            total_main_elapsed_time += elapsed_time
        energy_after = read_energy()
        print(f"Energy after: {energy_after}")
        print(f"Total main elapsed time: {total_main_elapsed_time}")
        avg_power = (energy_after - energy_before) / total_main_elapsed_time / NUMBER_OF_RUNS
        print(f"[RESULT] Average power for {command.split(' ')[1]} {command.split(' ')[4]}: {avg_power}")
