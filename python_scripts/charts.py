import matplotlib.pyplot as plt
from models import ExperimentResult


def time_powercap_scatter():
    pass

def time_batch_size_scatter(experiment_result: ExperimentResult):
    batch_size_cpu_gpu = [multiple_run_result.parameters.batch_size for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    execution_duration_cpu_gpu = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    batch_size_gpu = [multiple_run_result.parameters.batch_size for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    execution_duration_gpu = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    
    plt.plot(batch_size_cpu_gpu, execution_duration_cpu_gpu, label="CPU+GPU", marker='o')
    plt.plot(batch_size_gpu, execution_duration_gpu, label="GPU", marker='o')
    plt.xscale('log')
    plt.xlabel("batch size")
    plt.ylabel("time [s]")
    plt.legend()
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_batch_size_nodes_{experiment_result.experiment_result[0].parameters.number_od_nodes}.png')

def time_number_of_nodes_histogram():
    pass

def time_number_of_nodes_scatter():
    pass
