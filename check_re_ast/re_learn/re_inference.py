from .generator import generator


def re_infer(samples: list[str], POPULATION=100, GENERATION=20, match=None) -> list[tuple[int, str]]:
    return generator(samples, POPULATION=POPULATION, GENERATION=GENERATION, match=match)
