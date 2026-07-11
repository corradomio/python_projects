

def _split_url(url: str) -> tuple[str, str]:

    if url.startswith("file:///"):
        return "", url[8:]
    if url.startswith("file://"):
        return "", url[7:]

    pos = url.find("://")
    if pos == -1:
        return "", url

    url = url[pos+3:]
    pos = url.find("/")
    if pos == -1:
        return url, ""
    else:
        return url[:pos], url[pos + 1 :]


def protocol_of(url: str) -> str:
    pos = url.find("://")
    if pos == -1:
        return ""
    else:
        return url[:pos+3]


def normalize(path: str) -> str:
    path = path.replace("\\", "/")
    path = path.replace("//", "/")
    return path


def join_with(parent: str, child: str) -> str:
    assert not child.startswith("/")
    return normalize(parent + "/" + child)


def parent_of(path: str) -> str:
    if path == "/":
        return "/"
    pos = path.rfind("/")
    if pos == -1:
        return "/"
    else:
        return path[:pos]


def name_of(path: str) -> str:
    if path == "/": return ""
    pos = path.rfind("/")
    if pos == -1:
        return ""
    else:
        return path[pos + 1 :]

def stem_of(path: str) -> str:
    name = name_of(path)
    pos = path.rfind(".")
    return name if pos == -1 else name[:pos]


def suffix_of(path: str) -> str:
    name = name_of(path)
    pos = name.rfind(".")
    return "" if pos == -1 else path[pos:]
