import matplotlib.pyplot as plt
import pandas as pd
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

    plt.figure()
    plt.plot(power_cpu_gpu, execution_duration_cpu_gpu_avg, label="CPU+GPU avg", marker='o')
    plt.plot(power_gpu, execution_duration_gpu_avg, label="GPU avg", marker='o')
    plt.plot(power_cpu_gpu, execution_duration_cpu_gpu_min, label="CPU+GPU min", marker='o')
    plt.plot(power_gpu, execution_duration_gpu_min, label="GPU min", marker='o')
    plt.plot(power_cpu_gpu, execution_duration_cpu_gpu_max, label="CPU+GPU max", marker='o')
    plt.plot(power_gpu, execution_duration_gpu_max, label="GPU max", marker='o')
    plt.xlabel("powercap")
    plt.ylabel("time [s]")
    plt.legend()
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_power_cap_nodes_{experiment_result.experiment_result[0].parameters.number_od_nodes}.png')
    plt.close()


def time_batch_size_scatter(experiment_result: ExperimentResult):
    batch_size_cpu_gpu = [multiple_run_result.parameters.batch_size for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    execution_duration_cpu_gpu = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled]
    
    batch_size_gpu = [multiple_run_result.parameters.batch_size for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    execution_duration_gpu = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled]
    
    plt.figure()
    plt.plot(batch_size_cpu_gpu, execution_duration_cpu_gpu, label="CPU+GPU", marker='o')
    plt.plot(batch_size_gpu, execution_duration_gpu, label="GPU", marker='o')
    plt.xscale('log')
    plt.xlabel("batch size")
    plt.ylabel("time [s]")
    plt.legend()
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_batch_size_nodes_{experiment_result.experiment_result[0].parameters.number_od_nodes}.png')
    plt.close()

def time_number_of_nodes_bar(experiment_result: ExperimentResult):
    number_of_nodes = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result]
    # "CPU+GPU one stream", "CPU+GPU two streams", "GPU one stream", "GPU two streams"
    configuration = [f'{"CPU+" if multiple_run_result.parameters.cpu_enabled else ""}GPU {"one stream" if multiple_run_result.parameters.number_of_streams == 1 else "two streams"}' for multiple_run_result in experiment_result.experiment_result]
    execution_duration = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result]

    data = {
        'number of nodes': number_of_nodes,
        'Configuration': configuration,
        'Time (s)': execution_duration
    }

    df = pd.DataFrame(data)
    # pivot the data so that each "Number of nodes" has its own column for each configuration
    df_pivot = df.pivot(index='number of nodes', columns='Configuration', values='Time (s)')
    df_pivot.plot(kind='bar', width=0.8, figsize=(10, 6))
    plt.xlabel('number of nodes')
    plt.ylabel('time [s]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_nodes.png')

def time_number_of_nodes_scatter(experiment_result: ExperimentResult):
    nodes_cpu_gpu_one_stream = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1]
    execution_duration_cpu_gpu_one_stream = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1] 
    nodes_cpu_gpu_two_streams = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    execution_duration_cpu_gpu_two_streams = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    
    nodes_gpu_one_stream = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1]
    execution_duration_gpu_one_stream = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 1]
    nodes_gpu_two_streams = [multiple_run_result.parameters.number_od_nodes for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    execution_duration_gpu_two_streams = [multiple_run_result.average("execution_duration") for multiple_run_result in experiment_result.experiment_result if not multiple_run_result.parameters.cpu_enabled and multiple_run_result.parameters.number_of_streams == 2]
    
    plt.figure()
    plt.plot(nodes_cpu_gpu_one_stream, execution_duration_cpu_gpu_one_stream, label="CPU+GPU 1 stream", marker='o')
    plt.plot(nodes_cpu_gpu_two_streams, execution_duration_cpu_gpu_two_streams, label="CPU+GPU 2 streams", marker='o')
    plt.plot(nodes_gpu_one_stream, execution_duration_gpu_one_stream, label="GPU 1 stream", marker='o')
    plt.plot(nodes_gpu_two_streams, execution_duration_gpu_two_streams, label="GPU 2 streams", marker='o')
    plt.xlabel("number of nodes")
    plt.ylabel("time [s]")
    plt.legend()
    plt.savefig(f'{experiment_result.experiment_result[0].parameters.app_name}_time_nodes_nodes_{experiment_result.experiment_result[0].parameters.number_od_nodes}.png')
    plt.close()
