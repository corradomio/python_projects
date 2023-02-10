from path import Path


def as_path(p):
    if isinstance(p, (list, tuple)):
        return list(map(as_path, p))
    if isinstance(p, Path):
        return p
    if isinstance(p, str):
        return Path(p)
    else:
        raise ValueError(f"Unsupported type {type(p)}")