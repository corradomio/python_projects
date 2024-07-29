#
# Multiprocessing extensions
#
#   Set of functions for starting and monitoring Python subprocesses
#
#   1) each subprocess must have an unique name
#   2) it is not possible to start multiple processes with the same name
#   3) all processes are registered in the GLOBAL registry 'RUNNING_PROCESSES'
#   4) a process can have 'properties'
#
import multiprocessing as mp
from datetime import datetime
from typing import Union, Callable

PROCESS_ID = Union[int, str, tuple]


RUNNING_PROCESSES: dict[PROCESS_ID, dict] = {}


def is_running(name: str) -> bool:
    """
    Check if the process 'name' is running.

    Cleanup: if the proces is registered in the list of 'running' processed
    but it is terminated, it is removed from the registry

    :param name: process name
    """
    global RUNNING_PROCESSES
    assert isinstance(name, str)

    # cleanup: remove all terminated processes
    pnames = list(RUNNING_PROCESSES)
    for pname in pnames:
        # p = RUNNING_PROCESSES[pname]
        p = RUNNING_PROCESSES[pname]["process"]

        # skip the placeholder (description in 'start_process')
        if isinstance(p, bool):
            continue
        # if the process is terminated, delete it
        if p.is_alive():
            continue
        del RUNNING_PROCESSES[pname]
    # end

    return name in RUNNING_PROCESSES
# end


def process_properties(name: str) -> dict:
    """
    Get the process properties, enriched with status, start time, etc

    :param name: process name
    """
    global RUNNING_PROCESSES
    assert isinstance(name, str)

    if name not in RUNNING_PROCESSES:
        pinfo = dict(
            name=name,
            status="not_existent"
        )
    else:
        pinfo = RUNNING_PROCESSES[name]
    return pinfo
# end


def start_process(name: str, target: Callable, args=(), kwargs=None, pprops=None) -> str:
    """
    Start a subprocess with name 'name' and using the GLOBAL function
    'target'.

    :param name: process unique name
    :param target: function executed inside the process
    :param args: arguments passed to the target
    :param kwargs: named arguments passed to the target
    :param pprops: process properties

    :return: process name
    """
    global RUNNING_PROCESSES
    assert isinstance(name, str)

    # convert None into an empty dictionary
    kwargs = kwargs or {}
    pprops = ({} | pprops) or {}

    if is_running(name):
        return name
    else:
        # FAST placeholder to reduce the probability to start multiple processes
        # This because to create the python subprocess can requires several seconds
        RUNNING_PROCESSES[name] = pprops | {
            "name": name,
            "process": True,
            "status": "created"
        }

    # create a new process, put it in RUNNING_PROCESSES, and start it
    p = mp.Process(target=target, args=args, kwargs=kwargs)
    p.daemon = True
    RUNNING_PROCESSES[name] = RUNNING_PROCESSES[name] | {
        "start_time_": datetime.now(),
        "process": p,
        "status": "running"
    }
    p.start()

    return name
# end


def stop_process(name: str) -> str:
    """
    Stop/kill the process 'name'

    :param name: name of the process to kill
    :return: a simple string describing the result:

            'not_existent'      the proces doesn't exist
            'created'           the proces is 'created' but not yet registered
                                It is not possible to kill it because there is n 'Process'
                                object to kill
            'failed'            process killed
            'terminated'        process already terminated

    """
    global RUNNING_PROCESSES
    assert isinstance(name, str)

    if name not in RUNNING_PROCESSES:
        return "not_existent"

    # p can be a 'placeholder' (a bool value)
    p: Union[bool, mp.Process] = RUNNING_PROCESSES[name]["process"]
    if not isinstance(p,  mp.Process):
        return "created"

    if p.is_alive():
        p.kill()
        return "failed"
    else:
        return "terminated"
# end

