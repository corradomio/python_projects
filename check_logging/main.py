# This is a sample Python script.
import logging as log
import logging.config

# logging.basicConfig(level=logging.DEBUG,
#                     filename='myfirstlog.log',
#                     format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

logging.config.fileConfig('logging_config.ini')

import matplotlib.pyplot as plt

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    log.info("I'm an informational message.")
    log.debug("I'm a message for debugging purposes.")
    log.warning("I'm a warning. Beware!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
