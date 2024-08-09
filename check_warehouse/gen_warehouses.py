from random import random, randint
from stdlib import csvx

N =  200     # n of warehouses
M = 1000     # max number of items required/available
R = 0.125     # probability of warehouse requiring an item

HEADER = ["warehouse", "x", "y", "items"]
FILE = "warehouses.csv"


def main():

    data = []
    for i in range(N):
        name = f"w_{i+1:03}"
        x = random()
        y = random()
        r = random()
        w = (1 if r > R else -1) * randint(0, M)

        data.append([name, x, y, w])
    pass

    csvx.save_csv(FILE, data, header=HEADER)
# end


if __name__ == "__main__":
    main()
