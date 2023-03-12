
# for a in {'foo':1, 'bar':5, 'tic':6}:
#     print(a)
#     print(2*a)
#     if 'a' in 'Goofy':
#         print(1)
# print(6)
#

# def f(a):
#     b = 5
#     c = []
#     for i in range(b-1):
#         a, b = b, 2*a - b
#         c.append(str(b))
#     return c
#
# for a in range(0, 4):
#     print(f(a))


# print(10/0)

# num = int("45") * float("1.5")
# print(num)

# def f(L):
#     L = L + [5]
#     L.append(4)
# def g(D):
#     for e in D:
#         D[e] = e + e
# num = [1,3,2]
# dic = {1:'a', 2:'b', 3:'c'}
# print("Before:", num)
# print("Before:", dic)
# print("After:")
# f(num)
# print(num)
# g(dic)
# print(dic)

def f(m):
    s = str(m) + '2357'
    out = 0
    d = ''
    for c in s:
        n = int(c)
        out = (2*out + n) % 10
        d += str(out)
    return d

for a in range(5):
    print(f(a))