import os
import subprocess
import sys

from pathlib import Path

from models import RunParameters, SingleRunResult, ExperimentResult


def run_app(run_parameters: RunParameters) -> SingleRunResult:
    os.chdir(Path.home() / Path("parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib"))
    arguments = f"{run_parameters.number_of_streams} {run_parameters.powercap if run_parameters.powercap else ''}"
    command = f"./run_scripts/run-app {run_parameters.app_name} B {run_parameters.number_od_nodes} {arguments}"
    try:
        result = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
        print("Error Output:\n", e.output)
        sys.exit()
    return SingleRunResult.from_output(stdout=result.stdout, stderr=result.stderr)


def run_experiment(description: str, run_parameters: RunParameters, numer_of_runs: int) -> ExperimentResult:
    run_results = []
    for _ in range(numer_of_runs):
        run_results.append(run_app(run_parameters))
    return ExperimentResult(
        description=description,
        parameters=run_parameters,
        runs=run_results,
    )

def run_experiments():
    pass