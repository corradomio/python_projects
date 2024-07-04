# DON'T remove
import traceback
from logging import config, disable, shutdown, captureWarnings, getLevelName
import logging as log
from logging import DEBUG, INFO, WARN, WARNING, ERROR, CRITICAL, FATAL
import time


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def getLogger(name):
    return get_logger(name)


def get_logger(name):
    return Logger.getLogger(name)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    TIMEDELAY = 3

    # -----------------------------------------------------------------------

    # @staticmethod
    # def configure(**kwargs):
    #     log.basicConfig(**kwargs)

    # @staticmethod
    # def configure_using(fname: str, **kwargs):
    #     # log.basicConfig(filename=fname)
    #     log.config.fileConfig(fname, **kwargs)

    # @staticmethod
    # def configure_level(level=INFO):
    #     log.basicConfig(level=level)

    @staticmethod
    def getLogger(name):
        if not isinstance(name, str):
            name = type(name).__name__
        # logger = log.getLogger(name)
        # return Logger(logger)
        return Logger(name)

    # -----------------------------------------------------------------------

    # def __init__(self, logger):
    #     """
    #     :param log.Logger logger:
    #     """
    #     self._logger = logger
    #     self.timestamp = time.time()

    def __init__(self, name):
        """
        :param log.Logger logger:
        """
        self._name = name
        self._logger = None
        self.timestamp = time.time()

    @property
    def logger(self):
        if self._logger is None:
            self._logger = log.getLogger(self._name)
        return self._logger

    # -----------------------------------------------------------------------

    def setLevel(self, level):
        self.logger.setLevel(level)

    def isEnabledFor(self, level):
        return self.logger.isEnabledFor(level)

    # -----------------------------------------------------------------------

    # def is_debug_enabled(self):
    #     return self.isEnabledFor(DEBUG)
    #
    # def is_info_enabled(self):
    #     return self.isEnabledFor(INFO)
    #
    # def is_warning_enabled(self):
    #     return self.isEnabledFor(WARNING)
    #
    # def is_warn_enabled(self):
    #     return self.isEnabledFor(WARN)
    #
    # def is_error_enabled(self):
    #     return self.isEnabledFor(ERROR)
    #
    # def is_critical_enabled(self):
    #     return self.isEnabledFor(CRITICAL)
    #
    # def is_fatal_enabled(self):
    #     return self.isEnabledFor(FATAL)

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
        if self.isEnabledFor(DEBUG):
            self.debug(fmt.format(*args), **kwargs)

    def infof(self, fmt, *args, **kwargs):
        if self.isEnabledFor(INFO):
            self.info(fmt.format(*args), **kwargs)

    def warnf(self, fmt, *args, **kwargs):
        self.warn(fmt.format(*args), **kwargs)

    def warningf(self, fmt, *args, **kwargs):
        self.warning(fmt.format(*args), **kwargs)

    def errorf(self, fmt, *args, **kwargs):
        self.error(fmt.format(*args), **kwargs)

    def infot(self, msg, *args, **kwargs):
        if self.isEnabledFor(INFO):
            now = time.time()
            delta = now - self.timestamp
            if (delta) > self.TIMEDELAY:
                self.timestamp = now
                self.info(msg.format(*args), **kwargs)

    def debugt(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG):
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

    def full_error(self, e, fmt, *args, **kwargs):
        exc = traceback.format_exc()
        self.error(fmt.format(*args), **kwargs)
        self.error(f"... error type: {type(e)}\n{exc}")

# end


# ---------------------------------------------------------------------------
# loggers
# ---------------------------------------------------------------------------
# critical
# fatal
# error
# exception
# warning
# warn
# info
# debug
# log
# .

ROOT = "root"


def debug(msg, *args, **kwargs):
    Logger.getLogger(ROOT).debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    Logger.getLogger(ROOT).info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    Logger.getLogger(ROOT).warn(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    Logger.getLogger(ROOT).warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    Logger.getLogger(ROOT).error(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    Logger.getLogger(ROOT).fatal(msg, *args, **kwargs)


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

