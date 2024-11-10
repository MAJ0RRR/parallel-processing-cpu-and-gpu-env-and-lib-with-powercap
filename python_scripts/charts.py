import matplotlib.pyplot as plt
from models import ExperimentResult


def time_powercap_scatter(experiment_result: ExperimentResult):
    power_cpu_gpu = [multiple_run_result.parameters.powercap for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    execution_duration_cpu_gpu_avg = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    execution_duration_cpu_gpu_min = [multiple_run_result.min("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    execution_duration_cpu_gpu_max = [multiple_run_result.max("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]

    power_gpu = [multiple_run_result.parameters.powercap for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    execution_duration_gpu_avg = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    execution_duration_gpu_min = [multiple_run_result.min("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    execution_duration_gpu_max = [multiple_run_result.max("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]

    plt.plot(power_cpu_gpu, execution_duration_cpu_gpu_avg, label="CPU+GPU avg", marker='o')
    plt.plot(power_gpu, execution_duration_gpu_avg, label="GPU avg", marker='o')
    plt.plot(power_cpu_gpu, execution_duration_cpu_gpu_min, label="CPU+GPU min", marker='o')
    plt.plot(power_gpu, execution_duration_gpu_min, label="GPU min", marker='o')
    plt.plot(power_cpu_gpu, execution_duration_cpu_gpu_max, label="CPU+GPU max", marker='o')
    plt.plot(power_gpu, execution_duration_gpu_max, label="GPU max", marker='o')
    plt.xlabel("power cap")
    plt.ylabel("time [s]")
    plt.legend()
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_power_cap_nodes_{experiment_result.experiment_result[0].parameters.number_od_nodes}.png')


def time_batch_size_scatter(experiment_result: ExperimentResult):
    batch_size_cpu_gpu = [multiple_run_result.parameters.batch_size for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    execution_duration_cpu_gpu = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    
    batch_size_gpu = [multiple_run_result.parameters.batch_size for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
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

def time_number_of_nodes_scatter(experiment_result: ExperimentResult):
    nodes_cpu_gpu_one_stream = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1]
    execution_duration_cpu_gpu_one_stream = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1] 
    nodes_cpu_gpu_two_streams = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    execution_duration_cpu_gpu_two_streams = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    
    nodes_gpu_one_stream = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1]
    execution_duration_gpu_one_stream = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1]
    nodes_gpu_two_streams = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    execution_duration_gpu_two_streams = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]

    plt.plot(nodes_cpu_gpu_one_stream, execution_duration_cpu_gpu_one_stream, label="CPU+GPU 1 stream", marker='o')
    plt.plot(nodes_cpu_gpu_two_streams, execution_duration_cpu_gpu_two_streams, label="CPU+GPU 2 streams", marker='o')
    plt.plot(nodes_gpu_one_stream, execution_duration_gpu_one_stream, label="GPU 1 stream", marker='o')
    plt.plot(nodes_gpu_two_streams, execution_duration_gpu_two_streams, label="GPU 2 streams", marker='o')
    plt.xlabel("nodes")
    plt.ylabel("time [s]")
    plt.legend()
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_nodes_nodes_{experiment_result.experiment_result[0].parameters.number_od_nodes}.png')
