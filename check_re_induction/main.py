import sigmapie


class SigmaPie:

    def __init__(self, k=3, alphabet=None, polar="n"):
        if alphabet is None:
            alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 -_.:|/")

        self.k = k
        self.alphabet = alphabet
        self.polar = polar
        self.model = None

    def fit(self, X):
        self.model = sigmapie.SP(polar=self.polar)
        self.model.k = self.k
        self.model.alphabet = self.alphabet
        self.model.data = X
        self.model.learn()

    def predict(self, x):
        if isinstance(x, str):
            return self.model.scan(x)
        else:
            return [self.model.scan(t) for t in x]

    @property
    def grammar(self):
        return self.model.grammar


def main():
    sp = SigmaPie()
    sp.fit(["1", "2", "3"])

    print(sp.predict("5"))
    print(sp.predict("123"))

    sp = SigmaPie()
    sp.fit(["INV-530", "INV-534", "INV-946"])

    print(sp.predict("INV-539"))
    print(sp.predict("RES-999"))


if __name__ == "__main__":
    main()
