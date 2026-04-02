__all__ = [
    "getLogger",
    "get_logger",
    "debug",
    "info",
    "warn",
    "warning",
    "error",
    "fatal",
    "setFileLoggerFile",
]

# [DON'T remove]
import logging
from logging import config, disable, shutdown, captureWarnings, getLevelName
from logging import DEBUG, INFO, WARN, WARNING, ERROR, CRITICAL, FATAL
import traceback
import time
# [DON'T remove]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

TIMEDELAY = 3   # seconds

def getLogger(name):
    return get_logger(name)


def get_logger(name):
    return Logger.getLogger(name)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:

    # -----------------------------------------------------------------------

    @staticmethod
    def getLogger(name):
        if not isinstance(name, str):
            name = type(name).__name__
        return Logger(name)

    # -----------------------------------------------------------------------

    def __init__(self, name):
        """
        :param logging.Logger logger:
        """
        self._name = name
        self._logger = None
        self.timestamp = time.time()

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(self._name)
        return self._logger

    # -----------------------------------------------------------------------

    def setLevel(self, level):
        self.logger.setLevel(level)

    def isEnabledFor(self, level):
        return self.logger.isEnabledFor(level)

    def set_level(self, level):
        self.logger.setLevel(level)

    def is_enabled_for(self, level):
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
        self.timestamp = time.time()
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.timestamp = time.time()
        self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.timestamp = time.time()
        self.logger.warning(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.timestamp = time.time()
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self.logger.fatal(msg, *args, **kwargs)

    # -----------------------------------------------------------------------

    def debugf(self, fmt, *args, **kwargs):
        self.timestamp = time.time()
        if self.isEnabledFor(DEBUG):
            self.debug(fmt.format(*args), **kwargs)

    def infof(self, fmt, *args, **kwargs):
        self.timestamp = time.time()
        if self.isEnabledFor(INFO):
            self.info(fmt.format(*args), **kwargs)

    def warnf(self, fmt, *args, **kwargs):
        self.timestamp = time.time()
        self.warn(fmt.format(*args), **kwargs)

    def warningf(self, fmt, *args, **kwargs):
        self.timestamp = time.time()
        self.warning(fmt.format(*args), **kwargs)

    def errorf(self, fmt, *args, **kwargs):
        self.timestamp = time.time()
        self.error(fmt.format(*args), **kwargs)

    # -----------------------------------------------------------------------

    def warnt(self, msg, *args, **kwargs):
        if self.isEnabledFor(INFO):
            now = time.time()
            delta = now - self.timestamp
            if (delta) > TIMEDELAY:
                self.timestamp = now
                self.warning(msg.format(*args), **kwargs)

    def infot(self, msg, *args, **kwargs):
        if self.isEnabledFor(INFO):
            now = time.time()
            delta = now - self.timestamp
            if (delta) > TIMEDELAY:
                self.timestamp = now
                self.info(msg.format(*args), **kwargs)

    def debugt(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG):
            now = time.time()
            delta = now - self.timestamp
            if (delta) > TIMEDELAY:
                self.timestamp = now
                self.debug(msg.format(*args), **kwargs)

    def tprint(self, fmt, *args, force=False, **kwargs):
        # if not self.is_debug_enabled():
        #     return

        now = time.time()
        delta = now - self.timestamp
        if (delta) > TIMEDELAY or force:
            self.timestamp = now
            print(time.strftime("[%H:%M:%S] "), end="")
            print(fmt.format(*args), **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self.logger.exception(msg.format(*args), exc_info=exc_info, **kwargs)
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
# Replace File Logger
# ---------------------------------------------------------------------------

def setFileLoggerFile(filename: str):
    """
    Used to replace the file used by 'logging.FileLogger'.

    The code is:

        logging.config.fileConfig('logging_config.ini')
        loggingx.setFileLoggerFile("another.log")

    where 'logging_config.ini' has the following file handler configuration:

        [handler_logfile]
        class=FileHandler
        level=DEBUG
        formatter=fileFormatter
        kwargs={'filename':'app1.log', 'delay':True}

    where 'delay' is used to avoid to open the file if not necessary

    :param filename: new filename to use
    :return:
    """
    import os
    from logging import StreamHandler
    # 'logging.root' is a GLOBAL object
    handlers = logging.root.handlers

    for handler in handlers:
        if not isinstance(handler, logging.FileHandler):
            continue

        filename = os.fspath(filename)
        # keep the absolute path, otherwise derived classes which use this
        # can generate problems when the current directory changes
        handler.baseFilename = os.path.abspath(filename)

        if handler.stream is not None:
            handler.stream.close()
            StreamHandler.__init__(handler, handler._open())
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

