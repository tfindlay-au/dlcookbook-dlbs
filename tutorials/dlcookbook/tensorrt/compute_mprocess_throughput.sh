#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../../scripts/environment.sh
logparser=$DLBS_ROOT/python/dlbs/logparser.py
processor=$DLBS_ROOT/src/tensorrt/python/compute_mprocess_throughput.py
#------------------------------------------------------------------------------#
# In case if inference benchmark is done with multiple host threads, this script
# can be used to compute mean throughput. In general, there are three options to
# compute average in case of multi-process benchmarks:
# 1. Do nothing special and measure throughput as usual summing up throughput of
#    individual processes (groups of inference engines). Issue here is that
#    processes can start/stop at different moments and this approach can provide
#    too optimistic estimates.
# 2. Synch processes in the beginning and end of benchmarking process. This, as
#    opposed to Option 1, can provide pessimistic estimates, especially, for high
#    throughput models such as AlexnetOWT, where each second basically costs
#    thousands of images.
# 3. Extract batch times, remove certain number of first/last batch times and
#    compute throughput based on that value.
#
# This script computes mean time using approach #3 that is usually used in combination
# with #2.
#------------------------------------------------------------------------------#
logdir=./logs                   # Directory with log files.
tmpfile=./.tmp/results.json     # Temporary file with extracted results.
N=50                            # Number of first/last batches to throw away before
                                # computing mean throughput. Totally, 2*N points
                                # will be removed from consideration.

rm $tmpfile

params="exp.replica_batch,results.throughput,results.time_data,results.mgpu_effective_throughput,exp.num_gpus"
python ${logparser} ${logdir} --recursive --output_params ${params} --output_file $tmpfile

python ${processor} ${tmpfile} ${N}
