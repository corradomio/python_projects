import json

true = True
false = False
null = None


def load(file: str) -> dict:
    with open(file, mode="r") as fp:
        return json.load(fp)


def save(obj, file: str, **kwargs):
    if 'indent' not in kwargs:
        kwargs['indent'] = 4
    with open(file, mode="w") as fp:
        return json.dump(obj, fp, **kwargs)


json_load = load
json_save = save
