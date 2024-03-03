import logging as log
import time

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

DEBUG = log.DEBUG
INFO = log.INFO
WARN = log.WARN
WARNING = log.WARNING
ERROR = log.ERROR
CRITICAL = log.CRITICAL
FATAL = log.FATAL


def get_logger(name):
    return Logger.get_logger(name)


def getLogger(name):
    return log.getLogger(name)


def basic_config(**kwargs):
    return Logger.configure(**kwargs)


class Logger:
    DEBUG = log.DEBUG
    INFO = log.INFO
    WARN = log.WARN
    WARNING = log.WARNING
    ERROR = log.ERROR
    CRITICAL = log.CRITICAL
    FATAL = log.FATAL
    TIMEDELAY = 3

    # -----------------------------------------------------------------------

    @staticmethod
    def configure(**kwargs):
        log.basicConfig(**kwargs)

    @staticmethod
    def configure_using(fname):
        log.basicConfig(filename=fname)

    @staticmethod
    def configure_level(level=INFO):
        log.basicConfig(level=level)

    @staticmethod
    def get_logger(name):
        if not isinstance(name, str):
            name = type(name).__name__
        logger = log.getLogger(name)
        return Logger(logger)

    # -----------------------------------------------------------------------

    def __init__(self, logger):
        """
        :param log.Logger logger:
        """
        self.logger = logger
        self.timestamp = time.time()

    def is_debug_enabled(self):
        return self.logger.isEnabledFor(DEBUG)

    def is_info_enabled(self):
        return self.logger.isEnabledFor(INFO)

    # -----------------------------------------------------------------------

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    # -----------------------------------------------------------------------

    def debugf(self, fmt, *args, **kwargs):
        if self.is_debug_enabled():
            self.debug(fmt.format(*args), **kwargs)

    def infof(self, fmt, *args, **kwargs):
        if self.is_info_enabled():
            self.info(fmt.format(*args), **kwargs)

    def warnf(self, fmt, *args, **kwargs):
        self.warn(fmt.format(*args), **kwargs)

    def errorf(self, fmt, *args, **kwargs):
        self.error(fmt.format(*args), **kwargs)

    def debugt(self, msg, *args, **kwargs):
        if not self.is_debug_enabled():
            return

        now = time.time()
        delta = now - self.timestamp
        if (delta) > self.TIMEDELAY:
            self.timestamp = now
            self.debug(msg.format(*args), **kwargs)

    def tprint(self, fmt, *args, force=False, **kwargs):
        # if not self.is_debug_enabled():
        #     return

        now = time.time()
        delta = now - self.timestamp
        if (delta) > self.TIMEDELAY or force:
            self.timestamp = now
            print(time.strftime("[%H:%M:%S] "), end="")
            print(fmt.format(*args), **kwargs)
# end

