import time
from datetime import datetime


TIMEDELAY = 3  # seconds
TIMESTAMP = 0  # last timestamp


def tprint(*args, force=False, **nargs):
    """
    As "print" but it prints the message ONLY each TIMEDELAY seconds

    :param args: arguments passed to 'print'
    :param force: if to force the print
    :param nargs: named arguments passed to 'print'
    :return:
    """
    global TIMESTAMP

    now = time.time()
    if (now - TIMESTAMP) > TIMEDELAY or force:
        TIMESTAMP = now
        print(time.strftime("[%H:%M:%S] "), end="")
        print(*args, **nargs)
    if force:
        TIMESTAMP -= TIMEDELAY+1

# end


class Timing:

    class named:
        def __init__(self, t, n):
            self.t = t
            self.n = n

        def stop(self):
            self.t.stop(self.n)
        # end

    def __init__(self):
        self.timers = dict()
        """:type: dict[str, list]"""
        self.timers["total"] = [time.time(), time.time()]
    # end

    def start(self, timer="default", timers=None):
        """
        Start the timer or the list of timers
        :param str timer:
        :param list[str] timers:
        """
        if timers is None:
            timers = [timer]
        for timer in timers:
            self.timers[timer] = [time.time(), time.time()]
        return Timing.named(self, timer)
    # end

    def stop(self, timer="default", timers=None):
        """
        Stop the timer, the list fo timers, or all timers

        To stol all timers, 'timer' and 'timers' must be None

        :param str timer: timer to stop
        :param list[str] timers: list of timers to stop
        """
        if timer is None and timers is None:
            timers = list(self.timers.keys())
        elif timers is None:
            timers = [timer]
        for timer in timers:
            self.timers[timer][1] = time.time()
    # end

    def report(self):
        self.timers["total"][1] = time.time()

        for timer in self.timers:
            timing = self.timers[timer]
            delta = self._format(timing)
            print("Timer {0}: {1}".format(timer, delta))
        pass
    # end

    def _format(self, timing):
        s = timing[1] - timing[0]
        if s >= 0:
            s = int(s+0.5)
            seconds = s % 60
            s = s // 60
            minutes = s % 60
            s = s // 60
            hours = s % 24
            days = s // 24
            return "{0}:{1:02}:{2:02}:{3:02}".format(days, hours, minutes, seconds)
        else:
            return "not stopped"
    # end
# end


def delta_time(start: datetime, done: datetime):
    assert start <= done, "Start time must be before done time"
    seconds = int((done - start).total_seconds())
    if seconds < 60:
        return f"{seconds} s"
    elif seconds < 3600:
        s = seconds % 60
        m = seconds // 60
        return f"{m:02}:{s:02} s"
    else:
        s = seconds % 60
        seconds = seconds // 60
        m = seconds % 60
        h = seconds // 60
        return f"{h:02}:{m:02}:{s:02} s"
# end


