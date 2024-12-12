import os
import functools
from itertools import chain

from models import RunParameters, Experiment, ExperimentResult, MultipleRunResult, SingleRunResult
from charts import time_powercap_scatter, time_batch_size_scatter, time_number_of_nodes_bar, time_number_of_nodes_scatter
from experiments import run_experiment


PROBLEM_SIZE = 960_000_000
NUMBER_OF_RUNS = 10


def experiment_time_nodes(description: str, app_name: str, file_path: str | os.PathLike):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        batch_size=50000,
        powercap=None,
        problem_size=PROBLEM_SIZE,
    )
    experiment = Experiment(
        description=description,
        experiment_configurations=list(
            chain.from_iterable(
                [
                    [
                        common_run_parameters(cpu_enabled=False, number_of_streams=1, number_od_nodes=i),
                        common_run_parameters(cpu_enabled=True, number_of_streams=1, number_od_nodes=i),
                        common_run_parameters(cpu_enabled=False, number_of_streams=2, number_od_nodes=i),
                        common_run_parameters(cpu_enabled=True, number_of_streams=2, number_od_nodes=i),
                    ]
                for i in [1, 2, 4, 8, 16]
                ]
            )
        )
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)

def experiment_time_powercap(description: str, app_name: str, file_path: str | os.PathLike):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        batch_size=50000,
        number_od_nodes=16,
        number_of_streams=2,
        problem_size=PROBLEM_SIZE,
    )
    experiment = Experiment(
        description=description,
        experiment_configurations=list(
            chain.from_iterable(
                [
                    [
                        common_run_parameters(cpu_enabled=True, powercap=powercap),
                        common_run_parameters(cpu_enabled=False, powercap=powercap),
                    ]
                for powercap in range(300, 3001, 100)
                ]
            )
        )
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)


def experiment_time_batch_size(description: str, app_name: str, file_path: str | os.PathLike, number_of_nodes: int):
    common_run_parameters = functools.partial(
        RunParameters,
        app_name=app_name,
        number_od_nodes=number_of_nodes,
        number_of_streams=2,
        problem_size=PROBLEM_SIZE,
        powercap=None,
        initial_cpu_batch_size_scaling=100
    )
    experiment = Experiment(
        description=description,
        experiment_configurations=list(
            chain.from_iterable(
                [
                    [
                        common_run_parameters(cpu_enabled=True, batch_size=batch_size),
                        common_run_parameters(cpu_enabled=False, batch_size=batch_size),
                    ]
                for batch_size in [3_840_000, 960_000, 480_000, 120_000, 40_000, 12_800]
                ]
            )
        )
    )
    run_experiment(experiment_file_name=file_path, experiment=experiment, number_of_runs=NUMBER_OF_RUNS)


if __name__ == "__main__":
    # experiment_time_nodes(description="time(number_of_nodes) and number of streams", app_name="collatz", file_path="collatz_time_nodes.json")
    # experiment_time_powercap(description="time(powercap)", app_name="collatz", file_path="collatz_powercap_16_nodes.json")
    
    #experiment_time_batch_size(description="time(batch_size)", app_name="vecadd", file_path="vecadd_batch_size_16_nodes.json", number_of_nodes=16)
    #experiment_time_batch_size(description="time(batch_size)", app_name="vecadd", file_path="vecadd_batch_size_8_nodes.json", number_of_nodes=8)
    #experiment_time_batch_size(description="time(batch_size)", app_name="vecadd", file_path="cvecadd_batch_size_4_nodes.json", number_of_nodes=4)

    #experiment_time_batch_size(description="time(batch_size)", app_name="smoothing", file_path="smoothing_batch_size_4_nodes.json", number_of_nodes=4)
    #experiment_time_batch_size(description="time(batch_size)", app_name="smoothing", file_path="smoothing_batch_size_8_nodes.json", number_of_nodes=8)
    
    exp = ExperimentResult.from_file("../cudampilib/smoothing_batch_size_4_nodes.json")
    time_batch_size_scatter(exp)
    exp = ExperimentResult.from_file("../cudampilib/smoothing_batch_size_8_nodes.json")
    time_batch_size_scatter(exp)
    
    experiment_time_batch_size(description="time(batch_size)", app_name="smoothing", file_path="smoothing_batch_size_16_nodes.json", number_of_nodes=16)
    
    #experiment_time_batch_size(description="time(batch_size)", app_name="vecmaxdiv", file_path="vecmaxdiv_batch_size_16_nodes.json", number_of_nodes=16)
    #experiment_time_batch_size(description="time(batch_size)", app_name="vecmaxdiv", file_path="vecmaxdiv_batch_size_8_nodes.json", number_of_nodes=8)
    #experiment_time_batch_size(description="time(batch_size)", app_name="vecmaxdiv", file_path="vecmaxdiv_batch_size_4_nodes.json", number_of_nodes=4)
    
    # experiment_time_batch_size(description="time(batch_size)", app_name="patternsearch", file_path="patternsearch_batch_size_16_nodes.json", number_of_nodes=16)
    # experiment_time_batch_size(description="time(batch_size)", app_name="patternsearch", file_path="patternsearch_batch_size_8_nodes.json", number_of_nodes=8)
    # experiment_time_batch_size(description="time(batch_size)", app_name="patternsearch", file_path="patternsearch_batch_size_4_nodes.json", number_of_nodes=4)

    # experiment_time_batch_size(description="time(batch_size)", app_name="collatz", file_path="collatz_batch_size_16_nodes.json", number_of_nodes=16)
    # experiment_time_batch_size(description="time(batch_size)", app_name="collatz", file_path="collatz_batch_size_8_nodes.json", number_of_nodes=8)
    # experiment_time_batch_size(description="time(batch_size)", app_name="collatz", file_path="collatz_batch_size_4_nodes.json", number_of_nodes=4)
    
    
    exp = ExperimentResult.from_file("../cudampilib/smoothing_batch_size_16_nodes.json")
    time_batch_size_scatter(exp)
    '''

    exp = ExperimentResult.from_file("../cudampilib/collatz_batch_size_16_nodes.json")
    time_batch_size_scatter(exp)
    exp = ExperimentResult.from_file("../cudampilib/collatz_batch_size_8_nodes.json")
    time_batch_size_scatter(exp)
    exp = ExperimentResult.from_file("../cudampilib/collatz_batch_size_4_nodes.json")
    time_batch_size_scatter(exp)

    exp = ExperimentResult.from_file("../cudampilib/vecadd_batch_size_16_nodes.json")
    time_batch_size_scatter(exp)
    exp = ExperimentResult.from_file("../cudampilib/vecadd_batch_size_8_nodes.json")
    time_batch_size_scatter(exp)
    exp = ExperimentResult.from_file("../cudampilib/vecadd_batch_size_4_nodes.json")
    time_batch_size_scatter(exp)

    exp = ExperimentResult.from_file("../cudampilib/vecmaxdiv_batch_size_16_nodes.json")
    time_batch_size_scatter(exp)
    exp = ExperimentResult.from_file("../cudampilib/vecmaxdiv_batch_size_8_nodes.json")
    time_batch_size_scatter(exp)
    exp = ExperimentResult.from_file("../cudampilib/vecmaxdiv_batch_size_4_nodes.json")
    time_batch_size_scatter(exp)

    exp = ExperimentResult.from_file("../cudampilib/patternsearch_batch_size_16_nodes.json")
    time_batch_size_scatter(exp)
    exp = ExperimentResult.from_file("../cudampilib/patternsearch_batch_size_8_nodes.json")
    time_batch_size_scatter(exp)
    exp = ExperimentResult.from_file("../cudampilib/patternsearch_batch_size_4_nodes.json")
    time_batch_size_scatter(exp)

    '''

    # exp = ExperimentResult.from_file("../python_scripts/collatz_time_nodes.json")
    # time_number_of_nodes_bar(exp)
    # time_number_of_nodes_scatter(exp)
    # exp = ExperimentResult.from_file("../python_scripts/collatz_powercap_16_nodes.json")
    # time_powercap_scatter(exp)
    # exp = ExperimentResult.from_file("../python_scripts/collatz_batch_size_16_nodes.json")
    # time_batch_size_scatter(exp)