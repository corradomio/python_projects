from etime.lag import resolve_lag

print(resolve_lag({
    'input': [0, 2, 4, 6],
    'target': [1, 3, 5]

}))

print(resolve_lag(1))
print(resolve_lag(0))
print(resolve_lag(3))

print(resolve_lag({
    'type': 'day',
    'input': {
        'day': 2,
        'week': 2
    },
    'target': {
        'day': 3,
        'month': 1
    }

}))


print(resolve_lag({
    'type': 'hour',
    'input': {
        'day': 2,
        'week': 2
    },
    'target': {
        'day': 3,
        'month': 1
    }

}))


print(resolve_lag({
    'input': {
        24: 2,
        168: 2
    },
    'target': {
        24: 3,
        720: 1
    }
}))


print(resolve_lag({
    'type': 'second',
    'input': {
        'day': 2,
        'week': 2
    },
    'target': {
        'day': 3,
        'month': 1
    }

}))


print(resolve_lag({
    'length': 3,
    'current': False

}))


