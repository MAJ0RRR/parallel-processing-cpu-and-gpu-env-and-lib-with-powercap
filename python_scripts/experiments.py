import os
import subprocess
import sys

from pathlib import Path

from models import RunParameters, SingleRunResult, MultipleRunResult, ExperimentResult, Experiment


def single_app_run(run_parameters: RunParameters) -> SingleRunResult:
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


def multiple_app_runs(run_parameters: RunParameters, numer_of_runs: int) -> MultipleRunResult:
    run_results = []
    for _ in range(numer_of_runs):
        run_results.append(single_app_run(run_parameters))
    return MultipleRunResult(
        parameters=run_parameters,
        runs=run_results,
    )

def run_experiment(experiment_file_name: str, experiment: Experiment, number_of_runs: int):
    experiment_result = ExperimentResult(
        description=experiment.description,
        experiment_result=[],
    )

    for experiment_configuration in experiment.experiment_configurations:
        experiment_result.experiment_result.append(multiple_app_runs(run_parameters=experiment_configuration, numer_of_runs=number_of_runs))
    
    experiment_result.to_file(file_path=experiment_file_name)
