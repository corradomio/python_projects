import xml.etree.ElementTree as ET
from io import UnsupportedOperation
from typing import Union

import mitsuba as mi

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

TAG_IDS = {}
REF_ELEMENTS = {}


# ---------------------------------------------------------------------------
# load_scene_dict
# ---------------------------------------------------------------------------

def load_scene_dict(scene_xml: str, **kwargs) -> dict:
    # global TAG_IDS, REF_ELEMENTS
    TAG_IDS.clear()
    REF_ELEMENTS.clear()

    REF_ELEMENTS.update(kwargs)

    tree = ET.parse(scene_xml)
    root = tree.getroot()

    scene_dict = dict()
    _parse_tag(scene_dict, root)
    return scene_dict
# end


def load_scene(scene_xml: str, **kwargs):
    scene_dict = load_scene_dict(scene_xml, **kwargs)
    return mi.load_dict(scene_dict)
# end


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _resolve_default(value: str) -> str:
    if not value.startswith("$"):
        return value
    name = value[1:]
    assert name in REF_ELEMENTS, f"default value {value} not defined"
    return REF_ELEMENTS[name]
# end


def _float(x):
    x = _resolve_default(x)
    return float(x)

def _int(x) -> int:
    x = _resolve_default(x)
    return int(x)

def _str(x) -> str:
    x = _resolve_default(x)
    return str(x)

def _bool(x) -> bool:
    x = _resolve_default(x)
    return bool(x)

# ---------------------------------------------------------------------------

def _register_ref(xml: ET.Element, data):
    if "id" not in xml.attrib:
        return

    id = xml.attrib["id"]
    REF_ELEMENTS[id] = data


def _get_id(xml: ET.Element) -> str:
    if "id" in xml.attrib:
        return xml.attrib["id"]

    tag = xml.tag

    if tag not in TAG_IDS:
        TAG_IDS[tag] = 1
    id = TAG_IDS[tag]
    TAG_IDS[tag] += 1
    return f"{tag}_{id}"


def _get_type(xml: ET.Element) -> str:
    assert "type" in xml.attrib
    stype = xml.attrib["type"]
    return _resolve_default(stype)

# ---------------------------------------------------------------------------

def _parse_array(xml: ET.Element) -> Union[float, list[float]]:
    assert isinstance(xml, ET.Element)
    # <tag value="v"/>
    # <tag value="v1, v2, v3"/>
    # <tag value="v1  v2  v3"/>
    # <tag x="..", y="..", z=".."/>
    # ...
    if "x" in xml.attrib or "y" in xml.attrib or "z" in xml.attrib:
        return _parse_array_xyz(xml)
    else:
        return _parse_array_value(xml.attrib["value"])


def _parse_array_xyz(xml: ET.Element) -> list[float]:
    xvalue = 0
    yvalue = 0
    zvalue = 0
    if "x" in xml.attrib:
        xvalue = _float(xml.attrib["x"])
    if "y" in xml.attrib:
        yvalue = _float(xml.attrib["y"])
    if "z" in xml.attrib:
        zvalue = _float(xml.attrib["z"])
    return [xvalue, yvalue, zvalue]


def _parse_array_value(value: str | float | list[float]) -> Union[float, list[float]]:
    value = _resolve_default(value)
    if isinstance(value, (float, list)):
        return value

    value = value.strip()
    if "," in value:
        parts = value.split(",")
        values = list(map(lambda s: _float(s.strip()), parts))
        return values
    elif " " in value:
        parts = value.split(" ")
        parts = [p for p in parts if len(p) > 0]
        values = list(map(lambda s: _float(s.strip()), parts))
        return values
    else:
        return _float(value.strip())


def _parse_matrix_value(value: str) -> list[list[float]]:
    data = _parse_array_value(value)
    if len(data) == 9:
        return [
            data[0:3],
            data[3:6],
            data[6:]
        ]
    elif len(data) == 16:
        return [
            data[0:4],
            data[4:8],
            data[8:12],
            data[12:]
        ]
    else:
        raise ValueError(f"Matrix is not 3x3 or 4x4: {len(data)}")

# ---------------------------------------------------------------------------

def _parse_tag(data: dict, xml: ET.Element):
    assert isinstance(data, dict)
    assert isinstance(xml, ET.Element)
    assert xml.tag in TAG_PARSERS.keys(), f"Tag {xml.tag} unsupported"

    parse = TAG_PARSERS[xml.tag]
    parse(data, xml)


def _parse_scene(data: dict, xml: ET.Element):
    assert xml.tag == "scene"
    #   <scene>
    #   </scene>
    data["type"] = "scene"
    _parse_children(data, xml)


def _parse_children(data: dict, xml: ET.Element):
    nelts = len(xml)
    for i in range(nelts):
        child = xml[i]
        _parse_tag(data, child)


def _parse_shape(data: dict, xml: ET.Element):
    assert xml.tag == "shape"
    #   <shape type="">
    #       ...
    #   </shape>
    id = _get_id(xml)
    type = _get_type(xml)
    #
    shape = dict(type=type, id=id)
    _parse_children(shape, xml)
    #
    _register_ref(xml, shape)
    data[id] = shape
    pass


def _parse_integrator(data: dict, xml: ET.Element):
    assert xml.tag == "integrator"
    #   <integrator type="">
    #       <integer name="max_depth" value="5"/>
    #   </integrator>
    id = _get_id(xml)
    type = _get_type(xml)
    #
    integrator = dict(type=type, id=id)
    _parse_children(integrator, xml)
    #
    _register_ref(xml, integrator)
    data[id] = integrator
    pass


def _parse_sensor(data: dict, xml: ET.Element):
    assert xml.tag == "sensor"
    #   <sensor type="">
    #       ...
    #       <sampler type="..."> ... </sampler>
    #       <film type="..."> ... <film>
    #   </sensor>
    id = _get_id(xml)
    type = _get_type(xml)
    #
    sensor = dict(type=type, id=id)
    _parse_children(sensor, xml)
    #
    _register_ref(xml, sensor)
    data[id] = sensor
    pass


def _parse_sampler(data: dict, xml: ET.Element):
    assert xml.tag == "sampler"
    #   <sampler type="">
    #       ...
    #   </sampler>
    id = _get_id(xml)
    type = _get_type(xml)
    #
    sampler = dict(type=type, id=id)
    _parse_children(sampler, xml)
    #
    _register_ref(xml, sampler)
    data[id] = sampler
    pass


def _parse_film(data: dict, xml: ET.Element):
    assert xml.tag == "film"
    #   <film type="">
    #       ...
    #   </film>
    id = _get_id(xml)
    type = _get_type(xml)
    #
    film = dict(type=type, id=id)
    _parse_children(film, xml)
    #
    _register_ref(xml, film)
    data[id] = film
    pass


def _parse_emitter(data: dict, xml: ET.Element):
    assert xml.tag == "emitter"
    #   <emitter type="">
    #       ...
    #   </emitter>
    id = _get_id(xml)
    type = _get_type(xml)
    #
    emitter = dict(type=type, id=id)
    _parse_children(emitter, xml)
    #
    _register_ref(xml, emitter)
    data[id] = emitter
    pass


def _parse_bsdf(data: dict, xml: ET.Element):
    assert xml.tag == "bsdf"
    #   <bsdf type="">
    #       ...
    #   </bsdf>
    id = _get_id(xml)
    type = _get_type(xml)
    #
    bsdf = dict(type=type, id=id)
    _parse_children(bsdf, xml)
    #
    _register_ref(xml, bsdf)
    data[id] = bsdf
    pass


def _parse_rfilter(data: dict, xml: ET.Element):
    assert xml.tag == "rfilter"
    #   <rfilter type="">
    #       ...
    #   </rfilter>
    id = _get_id(xml)
    type = _get_type(xml)
    #
    rfilter = dict(type=type, id=id)
    _parse_children(rfilter, xml)
    #
    _register_ref(xml, rfilter)
    data[id] = rfilter
    pass


def _parse_ref(data: dict, xml: ET.Element):
    assert xml.tag == "ref"
    # <ref id="..."/>
    id = xml.attrib["id"]
    id = _resolve_default(id)
    # assert id in REF_ELEMENTS, f"Reference {id} not found"
    # return REF_ELEMENTS[id]
    data[id] = dict(type="ref",id=id)
    pass


def _parse_transform(data: dict, xml: ET.Element):
    assert xml.tag == "transform"
    #   <transform name="">
    #
    #   <transform>
    name = xml.attrib["name"]
    #
    t = mi.ScalarTransform4f()
    nt = len(xml)
    for i in range(nt):
        t = _apply_transform(t, xml[i])

    data[name] = t
    pass

# ---------------------------------------------------------------------------

def _parse_spectrum(data: dict, xml: ET.Element):
    assert xml.tag == "spectrum"
    raise UnsupportedOperation()

def _parse_texture(data: dict, xml: ET.Element):
    assert xml.tag == "texture"
    raise UnsupportedOperation()

def _parse_include(data: dict, xml: ET.Element):
    assert xml.tag == "include"
    raise UnsupportedOperation()

def _parse_alias(data: dict, xml: ET.Element):
    assert xml.tag == "alias"
    raise UnsupportedOperation()

def _parse_path(data: dict, xml: ET.Element):
    assert xml.tag == "path"
    raise UnsupportedOperation()


# ---------------------------------------------------------------------------

def _parse_boolean(data: dict, xml: ET.Element):
    assert xml.tag == "boolean"
    name = xml.attrib["name"]
    value = xml.attrib["value"]
    data[name] = _bool(value)
    pass


def _parse_integer(data: dict, xml: ET.Element):
    assert xml.tag == "integer"
    name = xml.attrib["name"]
    value = xml.attrib["value"]
    data[name] = _int(value)
    pass


def _parse_float(data: dict, xml: ET.Element):
    assert xml.tag == "float"
    name = xml.attrib["name"]
    value = xml.attrib["value"]
    data[name] = _float(value)
    pass


def _parse_string(data: dict, xml: ET.Element):
    assert xml.tag == "string"
    name = xml.attrib["name"]
    value = xml.attrib["value"]
    data[name] = _str(value)
    pass


def _parse_vector(data: dict, xml: ET.Element):
    assert xml.tag == "vector"
    # <vector name="" value=""/>
    # <vector name="" x="", y="", z=""/>
    name = xml.attrib["name"]
    point = _parse_array(xml)

    data[name] = point
    pass


def _parse_point(data: dict, xml: ET.Element):
    assert xml.tag == "point"
    # <point name="" value=""/>
    # <point name="" x="", y="", z=""/>
    name = xml.attrib["name"]
    point = _parse_array(xml)

    data[name] = point
    pass


def _parse_rgb(data: dict, xml: ET.Element):
    assert xml.tag == "rgb"
    # <rgb name="intensity" value="1"/>
    name = xml.attrib["name"]
    value = _parse_array(xml)

    data[name] = dict(
        type="rgb",
        value=value
    )
    pass


def _parse_default(data: dict, xml: ET.Element):
    assert xml.tag == "default"
    # <default name="" value=""/>
    name = xml.attrib["name"]
    value = xml.attrib["value"]

    if name not in REF_ELEMENTS:
        REF_ELEMENTS[name] = value
    else:
        print(f"Default: {name}={REF_ELEMENTS[name]} ({value})")
    pass

# ---------------------------------------------------------------------------

TAG_PARSERS = {
    "scene": _parse_scene,
    "shape": _parse_shape,
    "integrator": _parse_integrator,
    "sensor": _parse_sensor,

    "sampler": _parse_sampler,
    "film": _parse_film,
    "emitter": _parse_emitter,

    "boolean": _parse_boolean,
    "integer": _parse_integer,
    "float": _parse_float,
    "string": _parse_string,
    "vector": _parse_vector,
    "point": _parse_point,

    "rgb": _parse_rgb,
    "spectrum": _parse_spectrum,
    "rfilter": _parse_rfilter,

    "default": _parse_default,

    "transform": _parse_transform,

    "texture": _parse_texture,
    "bsdf": _parse_bsdf,

    "ref": _parse_ref,
    "include": _parse_include,
    "alias": _parse_alias,
    "path": _parse_path
}


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def _parse_rotate(t, xml: ET.Element):
    assert xml.tag == "rotate"
    # <rotate x="1" angle="5"/>
    # <rotate value="1,2,3" angle="5>
    axis = _parse_array(xml)
    angle = _float(xml.attrib["angle"])

    tr = mi.ScalarTransform4f().rotate(axis, angle)
    return tr @ t


def _parse_scale(t, xml: ET.Element):
    assert xml.tag == "scale"
    # <scale value="5"/>
    # <scale value="1,2,3">
    v = _parse_array(xml)
    ts = mi.ScalarTransform4f().scale(v)
    return ts @ t


def _parse_translate(t, xml: ET.Element):
    assert xml.tag == "translate"
    # <translate value="5"/>
    # <translate value="1,2,3">
    v = _parse_array(xml)
    tt = mi.ScalarTransform4f().translate(v)
    return tt @ t


def _parse_matrix(t, xml: ET.Element):
    assert xml.tag == "matrix"
    # <matrix value="-1 0 0 0 0 1 0 1 0 0 -1 6.8 0 0 0 1" />
    m = _parse_matrix_value(xml.attrib["value"])

    tm = mi.ScalarTransform4f(m)
    return tm @ t


def _parse_lookat(t, xml: ET.Element):
    assert xml.tag == "lookat"
    #   <lookat target="..." origin="..." up="..."/>
    target = _parse_array_value(xml.attrib["target"])
    origin = _parse_array_value(xml.attrib["origin"])
    up = _parse_array_value(xml.attrib["up"])

    tl = mi.ScalarTransform4f().look_at(origin=origin, target=target, up=up)
    return tl @ t

# ---------------------------------------------------------------------------

TAG_TRANSFORMS = {
    "translate": _parse_translate,
    "scale": _parse_scale,
    "rotate": _parse_rotate,
    "lookat": _parse_lookat,
    "matrix": _parse_matrix,
}


def _apply_transform(t, xml: ET.Element):
    assert isinstance(xml, ET.Element)
    assert xml.tag in TAG_TRANSFORMS.keys(), f"Transform {xml.tag} unsupported"

    transform = TAG_TRANSFORMS[xml.tag]
    return transform(t, xml)
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
