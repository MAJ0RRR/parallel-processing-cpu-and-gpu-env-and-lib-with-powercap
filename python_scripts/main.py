from experiments import run_experiment
from models import RunParameters, ExperimentResult


if __name__ == "__main__":
    run_parameters = RunParameters(
        app_name="collatz",
        cpu_enabled=True,
        number_of_streams=1,
        number_od_nodes=3,
        batch_size=0,
        powercap=None,
        problem_size=0,
    )
    experiment_result = run_experiment("test", run_parameters, 1)
    experiment_result.to_file("test.json")
    a = ExperimentResult.from_file("test.json")
    