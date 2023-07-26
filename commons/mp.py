import multiprocessing as mp
from typing import Union

PROCESS_ID = Union[int, str, tuple]


RUNNING_PROCESSES: dict[PROCESS_ID, mp.Process] = {}


def is_running(name):
    global RUNNING_PROCESSES
    assert isinstance(name, str)

    pnames = list(RUNNING_PROCESSES)
    for pname in pnames:
        p = RUNNING_PROCESSES[pname]
        # skip the placeholder
        if isinstance(p, bool):
            continue
        # if the process is terminated, delete it
        if p.is_alive():
            continue
        del RUNNING_PROCESSES[pname]
    # end
    return name in RUNNING_PROCESSES
# end


def start_process(name=None, target=None, args=(), kwargs={}) -> str:
    global RUNNING_PROCESSES
    assert isinstance(name, str)

    if is_running(name):
        return name
    else:
        # FAST placeholder to reduce the probability to start multiple processes
        RUNNING_PROCESSES[name] = True

    # create a new process, put it in RUNNING_PROCESSES, and start it
    p = mp.Process(target=target, args=args, kwargs=kwargs)
    p.daemon = True
    RUNNING_PROCESSES[name] = p
    p.start()

    return name
# end


def stop_process(name) -> str:
    global RUNNING_PROCESSES

    if name not in RUNNING_PROCESSES:
        return "not_existent"

    # p can be a 'placeholder' (a bool value)
    p: Union[bool, mp.Process] = RUNNING_PROCESSES[name]
    if not isinstance(p,  mp.Process):
        return "created"

    if p.is_alive():
        p.kill()
        return "failed"
    else:
        return "terminated"
# end


def typeid_of(pname) -> tuple[str, int]:
    name, id = pname.split("-")
    return name, int(id)

