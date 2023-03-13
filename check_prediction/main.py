from math import log2, sqrt
def sq(x): return x*x


def entropy(l: list[float]) -> float:
    def plogp(x: float) -> float:
        return 0. if x == 0 else -x * log2(x)
    return sum(plogp(e) for e in l)


def norm(l: list[float]) -> float:
    return sqrt(sum(sq(e) for e in l))


def classification_quality(classification: list[float], normalize=True, mode: str = 'entropy') -> float:
    c = len(classification)
    if normalize:
        t = sum(classification)
        if t == 0: return 0.
        classification = [e/t for e in classification]
    if mode == 'entropy':
        return 1 - entropy(classification)/log2(c)
    if mode == 'euclidean':
        t = 1/c*sqrt(c)
        return (norm(classification) - t)/(1 - t)
    else:
        raise ValueError(f"Invalid mode '{mode}'")
# end


