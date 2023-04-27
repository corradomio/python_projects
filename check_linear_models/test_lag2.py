from etime.lag import resolve_lag

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