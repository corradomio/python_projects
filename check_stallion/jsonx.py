import json

true = True
false = False
null = None


def json_load(file: str) -> dict:
    with open(file, mode="r") as fp:
        return json.load(fp)
