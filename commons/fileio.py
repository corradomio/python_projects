#
#
#

def str_from_file(filepath, encoding="utf-8"):
    """
    Return the content of a text file as a string
    :param str|Path filepath: 
    :param str encoding: file encoding
    :return str: the content of the file
    """
    with open(filepath, mode="r", encoding=encoding) as f:
        content = " ".join(f)
        # f.close() closed by with
    return content
