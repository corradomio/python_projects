import json
from typing import Any


class JSONEncoderEx(json.JSONEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default(self, o: Any) -> Any:
        clazz = type(o)
        return super().default(o)

    def encode(self, o: Any) -> str:
        return super().encode(o)

    def iterencode(self, o, _one_shot=False):
        return super().iterencode(o, _one_shot)

    def __call__(self, *, skipkeys=False, ensure_ascii=True,
            check_circular=True, allow_nan=True, sort_keys=False,
            indent=None, separators=None, default=None):
        self.skipkeys = skipkeys
        self.ensure_ascii = ensure_ascii
        self.check_circular = check_circular
        self.allow_nan = allow_nan
        self.sort_keys = sort_keys
        self.indent = indent
        if separators is not None:
            self.item_separator, self.key_separator = separators
        elif indent is not None:
            self.item_separator = ','
        if default is not None:
            self.default = default
        return self
# end


def write(obj, json_file: str, **kwargs):
    with open(json_file, mode="w") as fp:
        json.dump(obj, fp, cls=JSONEncoderEx(), **kwargs)


