import os
import subprocess
import sys

from pathlib import Path

from cudampi_dataclasses import RunParameters, SingleRunResult

MODE = "C 13 14 15" # "C {idx_in_hostfile_1} {idx_in_hostfile_2} ... {idx_in_hostfile_n}" (e.g. "C 1 7 13")

def run_app(run_parameters: RunParameters) -> SingleRunResult:
    os.chdir(Path.home() / Path("parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib"))
    arguments = f"A {run_parameters.number_of_streams} {run_parameters.powercap if run_parameters.powercap else ''}"
    command = f"./run_scripts/run-app {run_parameters.app_name} {MODE} {arguments}"
    try:
        result = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
        print("Error Output:\n", e.output)
        sys.exit()
    return SingleRunResult.from_output(stdout=result.stdout, stderr=result.stderr)