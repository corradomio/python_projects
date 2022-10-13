from multimethod import multimethod
# This is a sample Python script.


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


@multimethod
def print_hi(name: str):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, string:{name}')  # Press Ctrl+F8 to toggle the breakpoint.


@multimethod
def print_hi(name: int):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, int:{name}')  # Press Ctrl+F8 to toggle the breakpoint.


def main():
    m = 'PyCharm'
    print_hi(m)
    m = 100
    print_hi(m)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
