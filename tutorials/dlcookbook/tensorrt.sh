#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=tensorrt
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run TensorRT. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
# This example runs in a container.
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"tensorrt"'\
                           -Pexp.docker=true\
                           -Pexp.gpus='0'\
                           -Pexp.phase='"inference"'\
                           -Vexp.model='["resnet18"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorrt/${exp.model}.log"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title"
    python $parser ./$framework/*.log --output_params ${params}
fi
#------------------------------------------------------------------------------#
# Example: same experiment as above but runs in a host OS. I must run this as a root
# sudo .... Do not know why for now, related thread:
# https://devtalk.nvidia.com/default/topic/1024906/tensorrt-3-0-run-mnist-sample-error-assertion-engine-failed/
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"tensorrt"' -Pexp.docker=false\
                           -Pexp.gpus='0' -Pexp.phase='"inference"'\
                           -Vexp.model='["resnet18"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorrt/${exp.model}.log"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title"
    python $parser ./$framework/*.log --output_params ${params}
fi
#------------------------------------------------------------------------------#
# Example: this one runs TensorRT with several models and several batch sizes
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                   -Pexp.framework='"tensorrt"'\
                   -Pexp.docker=true\
                   -Pexp.gpus='0'\
                   -Pexp.log_file='"${BENCH_ROOT}/tensorrt/${exp.model}_${exp.effective_batch}.log"'\
                   -Vexp.model='["alexnet", "googlenet", "deep_mnist", "eng_acoustic_model", "resnet50", "resnet101", "resnet152", "vgg16", "vgg19"]'\
                   -Vexp.replica_batch='[2, 4]'\
                   -Pexp.phase='"inference"'\
                   -Pexp.num_warmup_batches=1\
                   -Pexp.num_batches=1
    params="exp.framework_title,exp.effective_batch,results.time,results.total_time,exp.model_title"
    python $parser ./$framework/*.log --output_params ${params}
fi
#------------------------------------------------------------------------------#
# Example: one approach to run TensorRT inference benchmarks on multiple GPUs.
# The idea is to launch multiple parallel benchmarks each running in its own process.
# This is exactly what this example does. It iterates over provided GPUs and launches
# DLBS instance for each GPU, then waits for all background child processes to complete.
# LIMITATIONS: if the goal is to benchmark scalability, make sure that each DLBS
# process will run exactly one benchmarks, same for all processes. This means that
# you must not provide variables to benchmarking suite - in command line case
# there must not present -V switches. If they present, they must contain single
# values (for instance, this is OK: -Vexp.model=["alexnet"] and this is not OK:
# -Vexp.model=["alexnet", "resnet18"]).
# Since the script will run multiple independent processes, each DLBS benchmark
# creates its own log file. In ideal case, no matter how many GPUs you use, the
# throughput for the same configuration should remain about the same.
# The easiest way to run it is to create your own copy of this file and make sure
# that the example below runs (if true; then) and change parameters.
if false; then
    # --------------------------------------------------------------------------
    # This parameters can be changed
    # Comma separated list of GPUs to use. The bash script will split this string
    # and will run DLBS benchmark in its own process for each GPU. FOr instance,
    # gpus="0" means run one benchmark on GPU 0 and gpus="3" means run one benchmark
    # on GPU 3. The gpu="0,2" means run two parallel benchmarks one on GPU 0 and
    # another one on GPU 2. Do not run more parallel benchmarks than number of GPUs
    # you have.
    gpus="0,1,2,3";
    # This is a standard model supported by TensorRT backend. See directory
    # $DLBS_ROOT/models for list of all supported models.
    model="resnet50"
    # Number of warmup batches that do not contribute to performance measurements.
    num_warmup_batches=50
    # Number of actual benchmark batches to process. Each process will independently
    # run this number of batches.
    num_batches=200
    # Batch size. Even though we can run multiple processes on different GPUs, this
    # is not a weak scaling compared to training benchmarks. These are inference
    # benchmarks and every benchmark will have exactly this effective batch size.
    batch_size=16
    # If this value is not -1, TensorRT backend will log the performance progress every
    # N-th processed batch (where N = report_frequency). If it is -1, the only performance
    # numbers is log file will be final numbers for entire benchmark (results.time and
    # results.throughput). If this value is, for instance, 20, then the TensorRT
    # backend will be computing performance statistics for every 20 batches and will
    # be writing them into a log file. Every such log entry will contain two keys:
    #    results.batch_progress=[mean_time, time_std, throughput]
    #    results.progress=[mean_time, time_std, throughput]
    # where 'results.batch_progress' is a performance stats for batch including
    # CPU <--> GPU transfer overhead and 'results.progress' is for inference only
    # excluding data transfers overhead. It's not pipelined now, so, for small
    # models the difference may be significant.
    # Values are arrays with three elements - mean batch time over last N batches,
    # its standard deviation and mean throughput.
    # It's useful to enable such logging for long lasting benchmarks. I am not sure
    # what the best value is, but if it's really long lasting benchmark with compute,
    # intensive  model, several hundreds sound reasonable.
    report_frequency=20
    # By default, all DLBS backends output to log files all individual batch times.
    # For long benchmarks, this may not be required since anyway progress statistics
    # can be used to estimate performance. Set this to "true" to disable outputting
    # individual batch times in the end (results.time_data)
    no_batch_times="true"
    # Make sure you know what you are doing if you want to change anything below
    # this line.
    # --------------------------------------------------------------------------

    rm -rf ./tensorrt

    gpus=(${gpus//,/ });
    for gpu in "${gpus[@]}"
    do
        python $script $action --log-level=$loglevel\
                               -Pexp.gpus=\"$gpu\"\
                               -Pexp.model=\"$model\"\
                               -Pexp.num_warmup_batches=$num_warmup_batches\
                               -Pexp.num_batches=$num_batches\
                               -Ptensorrt.report_frequency=$report_frequency\
                               -Ptensorrt.no_batch_times=$no_batch_times\
                               -Pexp.replica_batch=$batch_size\
                               -Pep.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
                               -Pexp.framework='"tensorrt"'\
                               -Pexp.docker=true\
                               -Pexp.phase='"inference"'\
                               -Pexp.log_file='"${BENCH_ROOT}/tensorrt/${exp.model}/${exp.gpus}.log"' &
    done
    # Wait for all background DLBS benchmarks to complete
    wait

    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title"
    python $parser ./tensorrt --recursive --output_params ${params}

fi
