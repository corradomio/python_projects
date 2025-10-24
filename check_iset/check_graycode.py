from stdlib import iset

N = 2

gc, bits = iset.gray_codes_list(N)

m = len(bits)
for i in range(m):
    print(iset.ibinlist(gc[i], N), bits[i])

# for S in iset.gen_gray_codes(N):
#     print(iset.ibinlist(S, N))

print(iset.iboolfun(-1, N, p=True))

# for S in iset.ibooltable(-1, 10):
#     print(iset.ibinlist(S, N))