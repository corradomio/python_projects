from stdlib.imathx import ilog2up

for i in range(128):
    print(f"{i:2} -> {ilog2up(i)}")