from collections import defaultdict


class CharComponent:
    def __init__(self):
        self.chars = set()
# end


class Automata:

    def __init__(self):
        self._states: dict[tuple[int, int], CharComponent] = defaultdict(lambda : CharComponent())
        self._states[(-1, -2)] = CharComponent()

    def add(self, sample):
        n = len(sample)
        s = -1
        for t in range(n):
            cc = self._states[(s, t)]

            s = t

# end


def load_data():
    with open("samples.txt") as fin:
        return [line for line in fin if not line.startswith("#")]


def main():
    samples = load_data()

    aut = Automata()

    for sample in samples:
        aut.add(sample)
    pass


if __name__ == "__main__":
    main()
