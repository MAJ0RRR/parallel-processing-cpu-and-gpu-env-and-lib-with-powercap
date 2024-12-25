import subprocess
import sys
import os
from pathlib import Path

ENERGY_UJ_FILE = "/sys/class/powercap/intel-rapl:0/energy_uj"
NUMBER_OF_RUNS = 5
PROBLEM_SIZE = 96_000_000
NUMBER_OF_STREAMS = 2
POWERCAP = 0
NUMBER_OF_NODES = 2
INITIAL_CPU_BATCH_SIZE_SCALING = 1


def read_energy(file_path: str = ENERGY_UJ_FILE):
    try:
        with open(file_path, 'r') as file:
            return float(file.read())
    except FileNotFoundError:
        print(f"Błąd: Plik '{file_path}' nie został znaleziony.")
    except PermissionError:
        print(f"Błąd: Brak uprawnień do odczytu pliku '{file_path}'.")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")


def run_script(command: str):
    try:
        result = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
        print("Error Output:\n", e.output)
        sys.exit()

    
if __name__ == "__main__":
    os.chdir(Path.home() / Path("parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib"))

    commands = [
        f"./run_scripts/run-app collatz B {NUMBER_OF_NODES} --cpu-enabled=0 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=1200000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling={INITIAL_CPU_BATCH_SIZE_SCALING}",
        f"./run_scripts/run-app collatz B {NUMBER_OF_NODES} --cpu-enabled=1 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=1200000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling={INITIAL_CPU_BATCH_SIZE_SCALING}",
        f"./run_scripts/run-app vecadd B {NUMBER_OF_NODES} --cpu-enabled=0 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=120000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling={INITIAL_CPU_BATCH_SIZE_SCALING}",
        f"./run_scripts/run-app vecadd B {NUMBER_OF_NODES} --cpu-enabled=1 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=120000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling={INITIAL_CPU_BATCH_SIZE_SCALING}",
        f"./run_scripts/run-app vecmaxdiv B {NUMBER_OF_NODES} --cpu-enabled=0 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=1200000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling={INITIAL_CPU_BATCH_SIZE_SCALING}",
        f"./run_scripts/run-app vecmaxdiv B {NUMBER_OF_NODES} --cpu-enabled=1 --number-of-streams={NUMBER_OF_STREAMS} --batch-size=1200000 --powercap={POWERCAP} --problem-size={PROBLEM_SIZE} --initial-cpu-batch-size-scaling={INITIAL_CPU_BATCH_SIZE_SCALING}",
    ]
    result_from_single_run = []
    
    for command in commands:
        print(f"[COMMAND] {command}")
        for _ in range(NUMBER_OF_RUNS):
            energy_before = read_energy()
            run_script(command=command)
            energy_after = read_energy()
            result_from_single_run.append(energy_after - energy_before)
        print(f"[RESULT] Average energy for {command.split(' ')[1]} {command.split(' ')[4]}: {sum(result_from_single_run) / len(result_from_single_run)}")
        print("\n")
