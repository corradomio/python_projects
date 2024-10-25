import time
import sys
from concurrent.futures import ThreadPoolExecutor as TP

print(sys.version)
print(sys._is_gil_enabled())

def task():
    n = 0
    for x in range(100_000_000):
        n+=x
    return n

with TP() as pool:
    start = time.perf_counter()
    results = [pool.submit(task) for _ in range(6)]

print("Elapsed time:", time.perf_counter() - start)
print ([r.result() for r in results])
