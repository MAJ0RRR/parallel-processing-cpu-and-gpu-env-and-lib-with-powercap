import subprocess
import sys
import os
import re
from pathlib import Path
from dataclasses import dataclass


APP_NAME = "collatz"
MODE = "C 13 14 15" # "C {des_nr_1} {des_nr_2} ... {des_nr_n}" (e.g. "C 1 7 13")
NR_OF_STREAMS = 1
POWER_CAP = None # None or int meaning Watts


@dataclass
class SingleRunResultData:
    execution_duration: float # in seconds
    energy_used: float # in Watts

    @staticmethod
    def from_output(stdout: str, stderr: str) -> "SingleRunResultData":
        return SingleRunResultData(
            execution_duration=SingleRunResultData.get_execution_duration_from_output(stdout=stdout, stderr=stderr), 
            energy_used=SingleRunResultData.get_energy_used_from_output(stdout=stdout, stderr=stderr),
        )

    @staticmethod
    def get_execution_duration_from_output(stdout: str, stderr: str):
        main_time_match = re.search(r'Main elapsed time=([\d.]+)', stderr)
        return float(main_time_match.group(1))

    @staticmethod
    def get_energy_used_from_output(stdout: str, stderr: str):
        return 0.0



def run_cudampilib_app(app_name: str, mode: str, nr_of_streams: int, power_cap: int | None):
    os.chdir(Path.home() / Path("parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib"))
    arguments = f"A {nr_of_streams} {power_cap if power_cap else ''}"
    command = f"./run_scripts/run-app {app_name} {mode} {arguments}"
    try:
        result = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
        print("Error Output:\n", e.output)
        sys.exit()
    return SingleRunResultData.from_output(stdout=result.stdout, stderr=result.stderr)


if __name__ == "__main__":
    print(run_cudampilib_app(app_name=APP_NAME, mode=MODE, nr_of_streams=NR_OF_STREAMS, power_cap=POWER_CAP))
