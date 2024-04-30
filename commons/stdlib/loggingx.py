import logging as log
from logging import DEBUG, INFO, WARN, WARNING, ERROR, CRITICAL, FATAL
from logging import config
import time


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_logger(name):
    return Logger.get_logger(name)


def getLogger(name):
    return Logger.get_logger(name)


def basic_config(**kwargs):
    return Logger.configure(**kwargs)


# def getLogger(name):
#     return log.getLogger(name)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
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

    def is_warning_enabled(self):
        return self.logger.isEnabledFor(WARNING)

    def is_warn_enabled(self):
        return self.logger.isEnabledFor(WARN)

    def is_error_enabled(self):
        return self.logger.isEnabledFor(ERROR)

    def is_critical_enabled(self):
        return self.logger.isEnabledFor(CRITICAL)

    def is_fatal_enabled(self):
        return self.logger.isEnabledFor(FATAL)

    # -----------------------------------------------------------------------

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self.logger.fatal(msg, *args, **kwargs)

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


# ---------------------------------------------------------------------------
# loggers
# ---------------------------------------------------------------------------

def debug(msg, *args, **kwargs):
    Logger.get_logger("main").debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    Logger.get_logger("main").info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    Logger.get_logger("main").warn(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    Logger.get_logger("main").warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    Logger.get_logger("main").error(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    Logger.get_logger("main").fatal(msg, *args, **kwargs)


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

