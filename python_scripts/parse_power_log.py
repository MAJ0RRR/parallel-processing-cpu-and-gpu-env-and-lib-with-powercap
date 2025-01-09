import re
import json
import statistics


# Max value of the energy counter
ENERGY_MAX = 262143328850

def parse_log_file(file_path):
    results = []
    current_command = None
    energy_before = None
    runs = []
    total_elapsed_time = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace

            # Match COMMAND
            command_match = re.match(r'^\[COMMAND\] (.+)', line)
            if command_match:
                # Save the previous command's results
                if current_command:
                    results.append({
                        "command": current_command,
                        "energy_before": energy_before,
                        "runs": runs,
                        "total_elapsed_time": total_elapsed_time
                    })
                # Start a new command
                current_command = command_match.group(1)
                energy_before = None
                runs = []
                total_elapsed_time = None
                continue

            # Match ENERGY BEFORE
            energy_before_match = re.match(r'^ENERGY BEFORE: ([\d.]+)', line)
            if energy_before_match:
                energy_before = float(energy_before_match.group(1))
                continue

            # Match RUN details
            run_index_match = re.match(r'^RUN: (\d+)', line)
            if run_index_match:
                continue  # Skip, we'll handle this with the next line

            after_run_match = re.match(r'^After run \d+\. Elapsed time ([\d.]+), energy used: ([\d.]+)', line)
            if after_run_match:
                elapsed_time = float(after_run_match.group(1))
                energy_used = float(after_run_match.group(2))
                
                # Adjust for overflow
                if runs:
                    while energy_used < runs[-1]['energy_used']:
                        energy_used += ENERGY_MAX
                
                runs.append({
                    "elapsed_time": elapsed_time,
                    "energy_used": energy_used
                })
                continue

            # Match Total main elapsed time
            total_elapsed_match = re.match(r'^Total main elapsed time: ([\d.]+)', line)
            if total_elapsed_match:
                total_elapsed_time = float(total_elapsed_match.group(1))
                continue

    # Append the last command's results
    if current_command:
        results.append({
            "command": current_command,
            "energy_before": energy_before,
            "runs": runs,
            "total_elapsed_time": total_elapsed_time
        })

    return results


def calculate_power(data):
    results = []

    # Process each group of runs
    for entry in data:
        command = entry["command"]
        runs = entry["runs"]
        total_elapsed_time = entry["total_elapsed_time"]
        energy_before = entry["energy_before"]/1e6

        # Calculate cumulative energy and time
        cumulative_energy = [run['energy_used']/1e6 for run in runs]
        cumulative_time = [run['elapsed_time'] for run in runs]

        # Calculate power for each run
        powers = []
        for i, (energy, time) in enumerate(zip(cumulative_energy, cumulative_time)):
            if i == 0:
                before = energy_before
            else:
                before = cumulative_energy[i - 1]
            if time > 0:
                power = (energy - before) / time  # Power = Energy / Time
                powers.append(power)

        # Compute stats for power
        avg_power = sum(powers) / len(powers) if powers else 0
        stddev_power = statistics.stdev(powers) if len(powers) > 1 else 0
        min_power = min(powers) if powers else 0
        max_power = max(powers) if powers else 0

        results.append({
            "command": command,
            "average_power": avg_power,
            "stddev_power": stddev_power,
            "min_power": min_power,
            "max_power": max_power,
            "total_elapsed_time": total_elapsed_time,
            "total_energy_consumption": cumulative_energy[-1] if cumulative_energy else 0
        })

    # Print results
    for result in results:
        print(f"Command: {result['command']}")
        print(f"  Average Power: {result['average_power']:.2f} W")
        print(f"  Stddev Power: {result['stddev_power']:.2f} W")
        print(f"  Min Power: {result['min_power']:.2f} W")
        print(f"  Max Power: {result['max_power']:.2f} W")
        print(f"  Total Elapsed Time: {result['total_elapsed_time']:.2f} seconds")
        print(f"  Total Energy Consumption: {result['total_energy_consumption']:.2f} J")
        print()

def main():
    file_path = "results.txt"  # Replace with your file path
    parsed_data = parse_log_file(file_path)
    calculate_power(parsed_data)

if __name__ == "__main__":
    main()
