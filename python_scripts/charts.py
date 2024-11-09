import matplotlib.pyplot as plt


def time_powercap_scatter():
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='green', marker='o')
    # plt.xscale('log')
    plt.xlabel("powercap [W]")
    plt.ylabel("time [s]")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig('time_powercap_scatter.png')

def time_batch_size_scatter():
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='green', marker='o')
    # plt.xscale('log')
    plt.xlabel("batch size")
    plt.ylabel("time [s]")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig('time_batch_size.png')

def time_number_of_nodes_histogram():
    pass

def time_number_of_nodes_scatter():
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='green', marker='o')
    # plt.xscale('log')
    plt.xlabel("number of nodes")
    plt.ylabel("time [s]")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig('time_number_of_nodes.png')
