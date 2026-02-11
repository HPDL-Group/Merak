#!/bin/bash

current_dir=`pwd`
export PYTHONPATH=$PYTHONPATH:$current_dir/process_monitor

manager_url=`hostname -I | awk '{print $1}'`

yhrun -N 6 -n 6 -p 3090 python process_monitor/procguard/worker_launcher.py \
        --slurm --gpu-count 1 \
        --manager-url http://$manager_url:5001 \
        --command "python -u examples/adacheck/run_bert.py "
