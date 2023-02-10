from typing import Union, Iterable
from path import Path


def as_path(p: Union[str, list[str]]) -> Union[Path, list[Path]]:
    """
    Convert p, a str or list[str], in a Path or a list[Path]
    :param p: path or list of paths
    :return:
    """
    if isinstance(p, (list, tuple)):
        return list(map(as_path, p))
    if isinstance(p, Path):
        return p
    if isinstance(p, str):
        return Path(p)
    else:
        raise ValueError(f"Unsupported type {type(p)}")


def read_text(file: str) -> str:
    """
    Read a file and return it as a single string
    :param file: file to read
    :return: the file content as string
    """
    with open(file, encoding="utf-8") as rdr:
        lines = [line for line in rdr]
    return "".join(lines)


def split_camel(word: str) -> list[str]:
    """
    Split a word, written using CamelCase, in multiple parts

    :param word: word to split
    :return: list of parts
    """
    # U+ | Ul+ | U+Ul+
    # state:
    #   0 : init
    #   1 : lower
    #   2 : upper
    #   3: end
    n = len(word)
    if n == 0:
        return []

    status = 0
    parts = []
    s = 0
    for i in range(n):
        c = word[i]
        # U+
        if 'A' <= c <= 'Z':
            # start
            if status == 0:
                status = 2
                continue
            # U+
            if status == 2:
                continue
            # _lU
            else:
                parts.append(word[s:i])
                s = i
                status = 2
                continue
        if 'a' <= c <= 'z':
            # start
            if status == 0:
                status = 1
                continue
            # l+
            if status == 1:
                continue
            # _Ul
            elif s == i-1:
                status = 1
                continue
            else:
                parts.append(word[s:i - 1])
                s = i-1
                status = 1
                continue
        else:
            continue
    parts.append(word[s:n])
    return parts
# end
