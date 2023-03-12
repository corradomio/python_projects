
# def g(l):
#     for e in l:
#         yield e+10
#
# for e in g([1,2,3]):
#     print(e)
#
# print(list(g([1,2,3,4])))


def h():
    j = 0;
    while True:
        yield j
        j = j+1


for e in h():
    if e > 10:
        break
    else:
        print(e)


print(next(h()))


