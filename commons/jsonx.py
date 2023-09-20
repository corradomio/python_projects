import json

true = True
false = False
null = None


def json_load(file: str) -> dict:
    with open(file, mode="r") as fp:
        return json.load(fp)


def json_save(obj, file: str):
    with open(file, mode="w") as fp:
        return json.dump(obj, fp)
