def lcs(S, T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set

# test 1
print("test 1")
ret = lcs('academy', 'abracadabra')
for s in ret:
    print(s)

# test 2
print("test 2")
ret = lcs('ababc', 'abcdaba')
for s in ret:
    print(s)
