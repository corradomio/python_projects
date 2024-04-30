# This is a sample Python script.

import stdlib.loggingx as logging


def main():
    logging.config.fileConfig('logging_config.ini')

    # logging.info("I'm an informational message.")
    # logging.debug("I'm a message for debugging purposes.")
    # logging.warning("I'm a warning. Beware!")
    # logging.error("I'm an error. Houch!")

    print("--base--", flush=True)
    log = logging.getLogger("base")
    log.info("I'm an informational message.")
    log.debug("I'm a message for debugging purposes.")
    log.warning("I'm a warning. Beware!")
    log.error("I'm an error. Houch!")

    print("--base.child--", flush=True)
    log = logging.getLogger("base.child")
    log.info("I'm an informational message.")
    log.debug("I'm a message for debugging purposes.")
    log.warning("I'm a warning. Beware!")
    log.error("I'm an error. Houch!")

    print("--base.child.leaf--", flush=True)
    log = logging.getLogger("base.child.leaf")
    log.info("I'm an informational message.")
    log.debug("I'm a message for debugging purposes.")
    log.warning("I'm a warning. Beware!")
    log.error("I'm an error. Houch!")

    print("--base.other.leaf--", flush=True)
    log = logging.getLogger("base.other.leaf")
    log.info("I'm an informational message.")
    log.debug("I'm a message for debugging purposes.")
    log.warning("I'm a warning. Beware!")
    log.error("I'm an error. Houch!")

    print("--done--", flush=True)
# end


if __name__ == '__main__':
    main()
