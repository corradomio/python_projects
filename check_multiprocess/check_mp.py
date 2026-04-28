import time
from random import randint
from multiprocessing import Process

def f(name):
    while True:
        print('[f] hello', name)
        time.sleep(randint(1, 5))

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()

    while True:
        print('[main] hello', "cicciopasticcio")
        time.sleep(randint(1, 5))

    p.join()
