#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=nvcnn
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run NVCNN TensorFlow. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
# Now, only support for 'training'
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pnvcnn.use_nccl=true\
                           -Pnvcnn.use_xla=false\
                           -Pexp.dtype='"float32"'\
                           -Pexp.num_warmup_batches=100\
                           -Pexp.num_batches=400\
                           -Pexp.replica_batch=16\
                           -Pexp.framework='"nvcnn"'\
                           -Pexp.gpus='0'\
                           -Vexp.model='["alexnet"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/${exp.framework}/${exp.model}.log"'\
                           -Pexp.docker=true\
                           -Pexp.phase='"inference"'\
                           -Pnvcnn.docker_image='"nvcr.io/nvidia/tensorflow:17.12"'
    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                             results.time results.throughput exp.model_title\
                                             exp.docker_image
fi

