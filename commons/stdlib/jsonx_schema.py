# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
# <schema> ::=
#   {
#       "type": [<schema>] | {"key": <schema>, ...} | <type>
#       "default": <default_value> | None,
#       "format": <datetime_format>,
#       "empty": <boolean>,
#       "mandatory": <boolean>
#       "range": [<min_value> | None, <max_value> | None],
#       "values": ["v1", ..."] | None,
#       "note": Any
#   }
#
# <type> ::=
#       "bool" | "boolean"
#       "int" | "integer"
#       "float" | "real"
#       "str" | "string" | "text"
#       "date" | "datetime"
#       None
#
#   "value": can be used to specify a list of valid values
#   "range": can be used to specify the valid range
#       IF the value is outside the valid range, it is replaced with "default" value, IF available
#       OTHERWISE
#   "values": alternative to "range". List of possible valid values
#   "default": default value to use when it is not possible to transform the value in the correct type
#       for example '' in int/float/boolean value
#   "empty": if the field of string type, it check if the empty string is a valid value
#   "mandatory": if the field must be present
#
# if a key in the value is not present in <schema>, it is removed
#

SCHEMA_TYPES = {
    None: None,
    '': None,
    'none': None,
    'null': None,

    bool: bool,
    "bool": bool,
    "boolean": bool,

    int: int,
    "int": int,
    "integer": int,

    float: float,
    "float": float,
    "double": float,
    "real": float,

    str: str,
    "str": str,
    "string": str,
    "text": str,

    datetime: datetime,
    "date": datetime,
    "datetime": datetime,
}

def _as_schema(schema)-> dict:
    if schema is None:
        pass
    elif isinstance(schema, list):
        schema = [
            _as_schema(e_schema)
            for e_schema in schema
        ]
    elif isinstance(schema, dict):
        schema = {
            k: _as_schema(schema[k])
            for k in schema
        }
    elif isinstance(schema, str):
        schema = {"type": SCHEMA_TYPES[schema]}
    elif schema in [bool, int, float, str, datetime]:
        schema = {"type": SCHEMA_TYPES[schema]}
    else:
        pass

    return schema
# end


def _validate(value, schema):
    if schema is None:
        return value

    value_type = type(value)

    if isinstance(schema, list) and isinstance(value, list):
        value = [
            _validate(e, schema[0])
            for e in value
        ]
    elif isinstance(schema, dict) and isinstance(value, dict):
        value = {
            k: _validate(value[k], schema[k])
            for k in value
            if k in schema
        }
    elif value_type == schema["type"]:
        pass
    elif value_type == str and schema["type"] in [int, float, bool]:
        try:
            schema_type = schema["type"]
            value = schema_type(value)
        except:
            value = schema.get("default", None)
    elif value_type == str and schema["type"] == str:
        pass
    elif value_type == str and len(value) > 0 and schema.get("type", None) in [datetime]:
        try:
            value = datetime.fromisoformat(value)
        except:
            value = schema.get("default", None)
    else:
        pass

    return value
# end


def validate(record: Union[dict,list], schema: Union[dict,list]):
    assert isinstance(record, (dict, list))
    assert isinstance(schema, (dict, list))
    assert type(record) == type(schema)

    schema = _as_schema(schema)
    record = _validate(record, schema)

    return record
# end
