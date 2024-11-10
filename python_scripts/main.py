from models import RunParameters, Experiment, ExperimentResult
from charts import time_batch_size_scatter
from experiments import run_experiment


if __name__ == "__main__":
    # experiments definitions
    # time(batch_size) GPU vs CPU+GPU 2 nodes
    collatz1 = Experiment(
        description="COLLATZ - time(batch_size) GPU vs CPU + GPU 2 nodes",
        experiment_configurations=[
            RunParameters(
                app_name="collatz",
                cpu_enabled=False,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=10,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=False,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=100,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=False,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=1000,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=False,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=10000,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=False,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=100000,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=True,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=10,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=True,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=100,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=True,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=1000,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=True,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=10000,
                powercap=None,
                problem_size=100,
            ),
            RunParameters(
                app_name="collatz",
                cpu_enabled=True,
                number_of_streams=2,
                number_od_nodes=2,
                batch_size=100000,
                powercap=None,
                problem_size=100,
            ),
        ]
    )
    
    # run experiments
    # run_experiment(experiment_file_name="collatz1", experiment=collatz1, number_of_runs=2)
    
    # draw charts
    experiment_result = ExperimentResult.from_file("/home/macierz/s184717/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/collatz1")
    time_batch_size_scatter(experiment_result=experiment_result)
    