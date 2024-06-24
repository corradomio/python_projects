# ---------------------------------------------------------------------------
# is_filesystem
# ---------------------------------------------------------------------------

def is_filesystem(url: str) -> bool:
    """
    Check if the url is a filesystem, sthat is, starting with 'file://' or '<disk>:'.
    It must have the form '<protocol>://<rest>' or '<disk>:<rest>'

    :param url: url to analyze
    :return: true if it is a filesystem url
    """
    # file://....
    # <disk>:....
    if url.startswith("file://") or len(url) > 2 and url[1] == ':':
        return True
    elif "://" in url:
        return False
    else:
        raise ValueError(f"Unsupported datasource '{url}'")


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
