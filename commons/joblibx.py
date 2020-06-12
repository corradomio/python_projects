import joblib as jl
import random as rnd
from typing import Iterable

delayed = jl.delayed
Memory = jl.Memory

#
# The original joblib library creates a process for each element in the list.
# It keeps n_job processes active in each moment AND it is able to use an iterable object
#
# This is a BIG problem if it is necessary to transfer a lot of data inside the process:
# the time to transfer the data can be greater of the execution time.
#
# This library extends 'joblib' in this way:
#
# 1) the iterable is converted into a list
# 2) the list is splitted into n_job parts
# 3) n_job processes execute SEQUENTIALLY each list's part
# 4) the results will be collected into a list as in the original library
#
# The implementation introduces some tricks:
#
# 1) because the time of execution can be depend on the element position inside the list,
#    the original list is shuffled, and the results reordered at the end
# 2) the list is NOT splitted in n_jobs parts, but in n_jobs*n_splits parts. This to obtain
#    a little better distribution of the computation inside the processes
#


class Parallel:
    
    def __init__(self, n_jobs=None):
        """
        :param int n_jobs: n of parallel processes to use. Can be None,0 and 1
        :param int n_splits: the list of calls will be divided in 'n_jobs*n_splits' parts
        """
        if isinstance(n_jobs, (tuple, list)):
            self.n_jobs = n_jobs[0]
            self.n_splits = n_jobs[1]
        elif isinstance(n_jobs, int):
            self.n_jobs = n_jobs
            self.n_splits = 2
        elif n_jobs is None:
            self.n_jobs = None
            self.n_splits = None
        else:
            raise ValueError("Invalid parameter n_job: " + n_jobs)

    def __call__(self, iterable: Iterable) -> Iterable:
        if self.n_splits in [None, 0, 1]:
            # call the original implementation
            return _call_joblib(self.n_jobs, iterable)
        if self.n_jobs in [None, 0, 1]:
            # call the sequential implementation
            return _call_sequential(iterable)
        else:
            # call the parallel implementation
            return _call_parallel(self.n_jobs, self.n_splits, iterable)
    # end
# end


def _call_joblib(n_jobs: int, iterable: Iterable) -> Iterable:
    dummy = Memory
    def _call(f, args, kwargs): return f(*args, **kwargs)
    return jl.Parallel(n_jobs=n_jobs)(delayed(_call)(f, args, kwargs) for f, args, kwargs in iterable)


def _call_sequential(iterable: Iterable) -> Iterable:
    return [f(*args, **kwargs) for f, args, kwargs in iterable]


def _call_parallel(n_jobs: int, n_splits: int, iterable: Iterable) -> Iterable:
    calls = list(iterable)          # convert the iterable into a list
    #                                 each element of the list is the tuple (function, *args, **kwargs)
    nc = len(calls)                 # n of calls
    ns = n_jobs*n_splits            # n of splits
    sz = nc//ns                     # split size

    indices = list(range(nc))       # indices
    rnd.shuffle(indices)            # shuffle the indices

    # select the indices for each split
    isplits = [indices[i*sz:i*sz+sz] for i in range(ns)]
    if ns*sz < nc:
        isplits.append(indices[ns*sz:])
        ns += 1

    # select the calls for each split
    csplits = [[calls[i] for i in isplit] for isplit in isplits]

    # execute each split in parallel
    collected = jl.Parallel(n_jobs=n_jobs)(delayed(_call_sequential)(csplit) for csplit in csplits)

    # collect the results in the correct order
    results = [None]*nc
    for i in range(len(collected)):
        isplit = isplits[i]
        collect = collected[i]
        for j in range(len(collect)):
            k = isplit[j]
            results[k] = collect[j]
    return results

