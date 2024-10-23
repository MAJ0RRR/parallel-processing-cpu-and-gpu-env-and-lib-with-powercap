from experiments.run_experiment import run_experiment
from cudampi_dataclasses import RunParameters


if __name__ == "__main__":
    run_parameters = RunParameters(
        app_name="collatz",
        cpu_enabled=True,
        number_of_streams=1,
        batch_size=0,
        powercap=None,
        problem_size=0,
    )
    experiment_result = run_experiment("DES lab 3 nodes",run_parameters, 1)
    experiment_result.to_file("test")
    
