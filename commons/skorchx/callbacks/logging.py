import skorch.callbacks.logging as skcl
import time


class PrintLog(skcl.PrintLog):
    """
    Extends skorch.callbacks.PrintLog adding a delay on the logs.
    To replace the default log it is enough to use:

        net.set_params(callbacks__print_log=sktorchx.callbacks.PrintLog(delay=3))
    """

    def __init__(self, delay=3, sink=print, **kwargs):
        super().__init__(sink=sink, **kwargs)
        self.delay = delay
        self.timestamp = -delay

    def on_epoch_end(self, net, **kwargs):
        now = time.time()
        if now - self.timestamp >= self.delay:
            self.timestamp = now
            super().on_epoch_end(net, **kwargs)
