from stdlib.is_instance import is_instance
import random

def randnorm(v: float):
    return random.normalvariate(0, v)

def map_process(b: float|list[float], v: float|list[float], init: float|list[float], n: int) -> list[float]:
    # X[t] = SUM[i=1,p, b[i]X[t-i] + eps[t]
    assert is_instance(b, float|list[float])
    assert is_instance(v, float|list[float])
    assert is_instance(init, float|list[float])
    assert is_instance(b, list[float]) or is_instance(v, list[float])

    if is_instance(b, list[float]):
        n = len(b)
    elif is_instance(v, list[float]):
        n = len(v)
    elif is_instance(init, list[float]):
        n = len(init)

    if not is_instance(b, list[float]):
        b = [b]*n
    if not is_instance(v, list[float]):
        v = [v]*n
    if not is_instance(init, list[float]):
        init = [init]*n

    values = [randnorm(v[i]) for i in range(n)]

    for i in range(n):
        x = 0
        for j in range(n):
            x += values[i+j]*b[j]
        x += randnorm(v[i])
        values.append(x)
    return values[-n:]


def ar_process():
    pass


def arma_process():
    pass


def arima_process():
    pass


def sarma_process():
    pass


def sarima_process():
    pass


def farima_process():
    pass



def main():
    data = map_process()
    pass


if __name__ == "__main__":
    main()
