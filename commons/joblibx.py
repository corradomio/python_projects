import random as rnd
from typing import Iterable
from typing import Tuple, Union

import joblib

delayed = joblib.delayed
Memory = joblib.Memory

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
# 2) the list is split into n_job parts
# 3) n_job processes execute SEQUENTIALLY each list's part
# 4) the results will be collected into a list as in the original library
#
# The implementation introduces some tricks:
#
# 1) because the time of execution can be depend on the element position inside the list,
#    the original list is shuffled, and the results reordered at the end
# 2) the list is NOT split in n_jobs parts, but in n_jobs*n_splits parts. This to obtain
#    a little better distribution of the computation inside the processes
#

#        Parameters
#         ----------
#         n_jobs: int, default: None
#             The maximum number of concurrently running jobs, such as the number
#             of Python worker processes when backend="multiprocessing"
#             or the size of the thread-pool when backend="threading".
#             If -1 all CPUs are used.
#             If 1 is given, no parallel computing code is used at all, and the
#             behavior amounts to a simple python `for` loop. This mode is not
#             compatible with `timeout`.
#             For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
#             n_jobs = -2, all CPUs but one are used.
#             None is a marker for 'unset' that will be interpreted as n_jobs=1
#             unless the call is performed under a :func:`~parallel_config`
#             context manager that sets another value for ``n_jobs``.
#         backend: str, ParallelBackendBase instance or None, default: 'loky'
#             Specify the parallelization backend implementation.
#             Supported backends are:
#
#             - "loky" used by default, can induce some
#               communication and memory overhead when exchanging input and
#               output data with the worker Python processes. On some rare
#               systems (such as Pyiodide), the loky backend may not be
#               available.
#             - "multiprocessing" previous process-based backend based on
#               `multiprocessing.Pool`. Less robust than `loky`.
#             - "threading" is a very low-overhead backend but it suffers
#               from the Python Global Interpreter Lock if the called function
#               relies a lot on Python objects. "threading" is mostly useful
#               when the execution bottleneck is a compiled extension that
#               explicitly releases the GIL (for instance a Cython loop wrapped
#               in a "with nogil" block or an expensive call to a library such
#               as NumPy).
#             - finally, you can register backends by calling
#               :func:`~register_parallel_backend`. This will allow you to
#               implement a backend of your liking.
#
#             It is not recommended to hard-code the backend name in a call to
#             :class:`~Parallel` in a library. Instead it is recommended to set
#             soft hints (prefer) or hard constraints (require) so as to make it
#             possible for library users to change the backend from the outside
#             using the :func:`~parallel_config` context manager.
#         return_as: str in {'list', 'generator'}, default: 'list'
#             If 'list', calls to this instance will return a list, only when
#             all results have been processed and retrieved.
#             If 'generator', it will return a generator that yields the results
#             as soon as they are available, in the order the tasks have been
#             submitted with.
#             Future releases are planned to also support 'generator_unordered',
#             in which case the generator immediately yields available results
#             independently of the submission order.
#         prefer: str in {'processes', 'threads'} or None, default: None
#             Soft hint to choose the default backend if no specific backend
#             was selected with the :func:`~parallel_config` context manager.
#             The default process-based backend is 'loky' and the default
#             thread-based backend is 'threading'. Ignored if the ``backend``
#             parameter is specified.
#         require: 'sharedmem' or None, default None
#             Hard constraint to select the backend. If set to 'sharedmem',
#             the selected backend will be single-host and thread-based even
#             if the user asked for a non-thread based backend with
#             :func:`~joblib.parallel_config`.
#         verbose: int, optional
#             The verbosity level: if non zero, progress messages are
#             printed. Above 50, the output is sent to stdout.
#             The frequency of the messages increases with the verbosity level.
#             If it more than 10, all iterations are reported.
#         timeout: float, optional
#             Timeout limit for each task to complete.  If any task takes longer
#             a TimeOutError will be raised. Only applied when n_jobs != 1
#         pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}
#             The number of batches (of tasks) to be pre-dispatched.
#             Default is '2*n_jobs'. When batch_size="auto" this is reasonable
#             default and the workers should never starve. Note that only basic
#             arithmetics are allowed here and no modules can be used in this
#             expression.
#         batch_size: int or 'auto', default: 'auto'
#             The number of atomic tasks to dispatch at once to each
#             worker. When individual evaluations are very fast, dispatching
#             calls to workers can be slower than sequential computation because
#             of the overhead. Batching fast computations together can mitigate
#             this.
#             The ``'auto'`` strategy keeps track of the time it takes for a
#             batch to complete, and dynamically adjusts the batch size to keep
#             the time on the order of half a second, using a heuristic. The
#             initial batch size is 1.
#             ``batch_size="auto"`` with ``backend="threading"`` will dispatch
#             batches of a single task at a time as the threading backend has
#             very little overhead and using larger batch size has not proved to
#             bring any gain in that case.
#         temp_folder: str, optional
#             Folder to be used by the pool for memmapping large arrays
#             for sharing memory with worker processes. If None, this will try in
#             order:
#
#             - a folder pointed by the JOBLIB_TEMP_FOLDER environment
#               variable,
#             - /dev/shm if the folder exists and is writable: this is a
#               RAM disk filesystem available by default on modern Linux
#               distributions,
#             - the default system temporary folder that can be
#               overridden with TMP, TMPDIR or TEMP environment
#               variables, typically /tmp under Unix operating systems.
#
#             Only active when backend="loky" or "multiprocessing".
#         max_nbytes int, str, or None, optional, 1M by default
#             Threshold on the size of arrays passed to the workers that
#             triggers automated memory mapping in temp_folder. Can be an int
#             in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
#             Use None to disable memmapping of large arrays.
#             Only active when backend="loky" or "multiprocessing".
#         mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, default: 'r'
#             Memmapping mode for numpy arrays passed to workers. None will
#             disable memmapping, other modes defined in the numpy.memmap doc:
#             https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
#             Also, see 'max_nbytes' parameter documentation for more details.


class Parallel:

    # Note: there are some extra parameters:
    #
    #   pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}
    #   batch_size: int or 'auto', default: 'auto'
    
    def __init__(self, n_jobs: Union[None, int, Tuple[int, int]] = None, **kwargs):
        """
        :param int n_jobs: n of parallel processes to use.
            Can be None, 0, 1, or a integer greater than 1
            For None, 0, 1, it is used a "sequential" approach.
            It can be a tuple of 2 values:

                (n_jobs, n_splits)

            in this case, the list is subdivided in n_jobs*n_splits parts
            and submitted to 'n_job' jobs
        """
        n_splits = 1
        if isinstance(n_jobs, (list, tuple)):
            if len(n_jobs) >= 2:
                n_splits = n_jobs[1]
                n_jobs = n_jobs[0]
            elif len(n_jobs) == 1:
                n_jobs = n_jobs[0]
            else:
                n_jobs = 1
        # end
        assert n_jobs is None or isinstance(n_jobs, int) and n_jobs >= 0
        assert n_splits is None or isinstance(n_splits, int) and n_splits > 0

        self.n_jobs = 1 if n_jobs is None or n_jobs == 0 else n_jobs
        self.n_splits = 1 if n_splits is None else n_splits
        self.kwargs = kwargs
        pass
    # end

    def __call__(self, iterable: Iterable) -> Iterable:
        if self.n_jobs < 2:
            # call the sequential implementation
            return _call_sequential(iterable)
        elif self.n_splits == 0:
            # call the original implementation
            return _call_joblib(self.n_jobs, iterable, self.kwargs)
        else:
            # call the parallel implementation
            return _call_parallel(self.n_jobs, self.n_splits, iterable)
    # end
# end


def _call_sequential(iterable: Iterable) -> Iterable:
    return [f(*args, **kwargs) for f, args, kwargs in iterable]


def _call(f, args, kwargs):
    return f(*args, **kwargs)


def _call_joblib(n_jobs: int, iterable: Iterable, kwargs) -> Iterable:
    dummy = Memory
    return joblib.Parallel(n_jobs=n_jobs, **kwargs)(delayed(_call)(f, args, kwargs) for f, args, kwargs in iterable)


def _call_parallel(n_jobs: int, n_splits: int, iterable: Iterable) -> Iterable:
    calls = list(iterable)          # convert the iterable into a list
    #                                 each element of the list is the tuple (function, *args, **kwargs)
    nc = len(calls)                 # n of calls
    ns = n_jobs*n_splits            # n of splits
    sz = (nc + ns - 1)//ns                     # split size

    indices = list(range(nc))       # indices
    rnd.shuffle(indices)            # shuffle the indices

    # select the indices for each split
    isplits = [indices[i:min(i+sz, nc)] for i in range(0, nc, sz)]

    # select the calls for each split
    csplits = [[calls[i] for i in isplit] for isplit in isplits]

    # execute each split in parallel
    # print("... parallel")
    collected = joblib.Parallel(n_jobs=n_jobs)(delayed(_call_sequential)(csplit) for csplit in csplits)

    # collect the results in the correct order
    results = [None]*nc
    for i in range(len(collected)):
        isplit = isplits[i]
        collect = collected[i]
        for j in range(len(collect)):
            k = isplit[j]
            results[k] = collect[j]
        # end
    # end
    return results
# end
