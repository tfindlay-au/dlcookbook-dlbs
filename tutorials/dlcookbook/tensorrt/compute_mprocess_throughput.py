
import json

fname='./.tmp/results.json'   # Input file name with benchmark results
N=50                          # Remove this number of items from start/end.


with open(fname, 'r') as fobj:
    benchmarks = json.load(fobj)

benchmarks = benchmarks['data']
# The 'benchmarks' is a dictionary with single key 'data'. It is an array of results
# of individual benchmarks. Each array is a dictionary containing the following
# keys:
#    exp.replica_batch     Per-GPU batch size
#    results.throughput    Throughput computed by a benchmarking suite
#    results.time_data     Duration in milliseconds of individual batches.

adjusted_mean_time = 0
ngpus = 0
for (i,benchmark) in enumerate(benchmarks):
    times = benchmark['results.time_data'][N:-N]
    adjusted_mean_time += sum(times) / len(times)
    ngpus += benchmark['exp.num_gpus']
    print("Benchmark %d (per GPU perf): own throughput %f, effective throughput %f" % (i, benchmark['results.throughput'], benchmark['results.mgpu_effective_throughput']))

adjusted_mean_time = adjusted_mean_time /  len(benchmarks)
adjusted_throughput = ngpus * 1000.0 * benchmark['exp.replica_batch'] / adjusted_mean_time
print("Adjusted throughput: %f" % adjusted_throughput)
