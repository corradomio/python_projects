import numpy as np
from stdlib.jsonx import load


def required_parts():
    requests: dict = load("data/requests_available_uk.json")["requests"]
    print(len(requests))

    p131 = []
    for r in requests.values():
        if "000131" in r:
            r131 = r["000131"]
            p131.append(r131)

    p131 = np.array(p131)
    print(len(p131))
    print(p131.mean(), p131.std())


def parts_in_stock():
    requests: dict = load("data/requests_available_uk.json")["requests"]
    print(len(requests))

    p131 = []
    for r in requests.values():
        if "000131" in r:
            r131 = r["000131"]
            p131.append(r131)

    p131 = np.array(p131)
    print(len(p131))
    print(p131.mean(), p131.std())


def main():
    required_parts()
    parts_in_stock()



if __name__ == "__main__":
    main()

