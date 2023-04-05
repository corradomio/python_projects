from multiprocessing import Process

def f(name, i):
    print('hello', name, i)


if __name__ == '__main__':
    pl = []
    for i in range(100):
        p = Process(target=f, args=('bob',i))
        p.start()
        pl.append(p)

    for p in pl:
        p.join()
    print("done")


