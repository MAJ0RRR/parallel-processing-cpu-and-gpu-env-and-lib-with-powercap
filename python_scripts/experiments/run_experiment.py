from cudampi_dataclasses import ExperimentResult, RunParameters
from experiments.run_app import run_app

def run_experiment(description: str, run_parameters: RunParameters, numer_of_runs: int) -> ExperimentResult:
    run_results = []
    for _ in range(numer_of_runs):
        run_results.append(run_app(run_parameters))
    return ExperimentResult(
        description=description,
        parameters=run_parameters,
        runs=run_results,
    )