#
#
from sktimex.lag import resolve_lag

def atest1():
    print(resolve_lag(5, current=False))
    print(resolve_lag((3, 5), current=False))
    print(resolve_lag({
        'input': {
            1: 3
        },
    }, current=False))
    print(resolve_lag({
        'target': {
            1: 3
        },
    }, current=False))
    print(resolve_lag({
        'target': {
            1: 3
        },
    }, current=True))
    print(resolve_lag({
        'current': True,
        'target': {
            1: 3
        },
    }, current=False))
    print(resolve_lag({
        'current': True,
        'target': {
            1: 3
        },
    }, current=None))

    print(resolve_lag({
        'current': False,
        'input': {
            1: 3,
            7: 5
        },
        'target': {
            1: 5,
            7: 2
        }
    }))


def atest2():
    # print(resolve_lag(5, current=False))
    # print(resolve_lag((3, 5), current=False))
    print(resolve_lag({
        'type': 'S',
        'input': {
            'M': 3
        },
        'current': True
    }))

    print(resolve_lag({
        'type': 'S',
        'target': {
            'H': 3
        },
    }, current=False))

    print(resolve_lag({
        'type': 'M',
        'target': {
            'H': 3
        },
    }, current=True))

    print(resolve_lag({
        'type': 'm',
        'current': True,
        'target': {
            'q': 3
        },
    }, current=False))

    print(resolve_lag({
        'type': 'S',
        'current': True,
        'target': {
            'M': 3
        },
    }, current=None))

    print(resolve_lag({
        'current': False,
        'input': {
            'day': 3,
            'week': 5
        },
        'target': {
            'day': 5,
            'week': 2
        }
    }))
    pass


def main():
    # atest1()
    atest2()


if __name__ == "__main__":
    main()