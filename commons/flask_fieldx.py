from flask_restx import fields, Api


def Union(*models) -> fields.Nested:
    if isinstance(models[0], dict):
        mdict = models[0]
        return fields.Nested(mdict)
    mdict = {}
    index = 0
    for model in models:
        mdict[str(index)] = model
        index += 1
    # end
    return fields.Nested(mdict)


def List(model) -> fields.List:
    return fields.List(fields.Nested(model))


# ---------------------------------------------------------------------------


class BytesMixin(object):
    __schema_type__ = "bytes"

    def __init__(self, *args, **kwargs):
        super(BytesMixin, self).__init__(*args, **kwargs)


class Bytes(BytesMixin, fields.Raw):
    """
    Marshal a value as a string.
    """

    def __init__(self, *args, **kwargs):
        super(BytesMixin, self).__init__(*args, **kwargs)

    def format(self, value):
        try:
            return str(value)
        except ValueError as ve:
            raise fields.MarshallingError(ve)
