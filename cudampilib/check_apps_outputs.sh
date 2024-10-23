#!/bin/bash

apps=("vecadd" "vecmaxdiv" "patternsearch" "collatz")

gpu_cpu_branch="adjust_apps_to_async_cpu_streams"
gpu_branch="adjust_apps_to_async_cpu_streams_GPU"

gpu_cpu_log_base="_logs_cpugpuasyncfull.log"
gpu_log_base="_logs_gpu.log"
logs_dir="logs/"

repo_base_dir="parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib"

cd "$repo_base_dir" || { echo "Failed to navigate to $repo_base_dir, run this from outside of repository."; exit 1; }
pwd

for app in "${apps[@]}"
do
    echo "APP: $app"
    for (( slaves=1; slaves<=3; slaves++ ))
    do   
        machines=$((slaves + 1))
        for (( streams=1; streams<=2; streams++ ))
        do
            echo -e "\tSlave count: $slaves"
            echo -e "\tStream count: $streams"

            git checkout $gpu_cpu_branch > /dev/null 2>&1
            ./compile > /dev/null 2>&1
            ./run_scripts/run-app "$app" B "$machines" "$streams"

            git checkout "$gpu_branch" > /dev/null 2>&1
            ./compile > /dev/null 2>&1
            ./run_scripts/run-app "$app" B "$machines" "$streams"

            gpu_cpu_log="${logs_dir}${app}${gpu_cpu_log_base}"
            gpu_log="${logs_dir}${app}${gpu_log_base}"

            diff "$gpu_cpu_log" "$gpu_log"

            if [ $? -eq 1 ]; then
                echo "Only header differ - correct!"
            else
                echo "Output is different - wrong!"
            fi

            rm "$gpu_cpu_log" > /dev/null 2>&1
            rm "$gpu_log" > /dev/null 2>&1
        done
    done
done