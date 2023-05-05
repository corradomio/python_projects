# Multiprocessing extensions

import multiprocessing as mp
from typing import Union

PROCESS_ID = Union[int, str, tuple]


RUNNING_PROCESSES: dict[PROCESS_ID, mp.Process] = {}


def start_process(name=None, target=None, args=(), kwargs={}) -> str:
    global RUNNING_PROCESSES

    if name is None:
        name = (str(target),) + tuple(args)

    # delete not alive processed
    for pname in list(RUNNING_PROCESSES):
        if not RUNNING_PROCESSES[pname].is_alive():
            del RUNNING_PROCESSES[pname]

    # if the process is already running, skip it
    if name in RUNNING_PROCESSES:
        return name

    # create a new process, put it in RUNNING_PROCESSES, and start it
    p = mp.Process(target=target, args=args, kwargs=kwargs)
    p.daemon = True
    RUNNING_PROCESSES[name] = p
    p.start()

    return name
# end


def stop_process(name):
    global RUNNING_PROCESSES

    if name not in RUNNING_PROCESSES:
        return

    p: mp.Process = RUNNING_PROCESSES[name]
    if isinstance(p,  mp.Process) and p.is_alive():
        p.kill()
# end
