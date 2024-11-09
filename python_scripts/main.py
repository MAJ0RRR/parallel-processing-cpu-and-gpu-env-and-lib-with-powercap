from models import RunParameters, Experiment
from experiments import run_experiment


if __name__ == "__main__":
    experiment_time_power = Experiment(
        description="Comparison of the dependence of application execution time on batch size for 2, 4, 16 nodes between GPU and GPU+CPU",
        experiment_configurations=[
            RunParameters(
                app_name="collatz",
                cpu_enabled=False,
                number_of_streams=1,
                number_od_nodes=2,
                batch_size=10,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=False,
                number_of_streams=1,
                number_od_nodes=2,
                batch_size=100,
                powercap=None,
                problem_size=100,
            ),
        ]
    )

    run_experiment(experiment_file_name="experiment_1", experiment=experiment_time_power, number_of_runs=1)
    
