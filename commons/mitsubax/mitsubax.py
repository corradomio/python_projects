import json
import xml.etree.ElementTree as ET
from io import UnsupportedOperation
from math import sqrt, asin, pi
from pathlib import Path
from typing import Callable, Any, cast, Optional, Union, Literal

import drjit as dr
import mitsuba as mi
import numpy as np

from .sensorx import ForwardSensor

TAG_FUNCTION = Callable[[dict, ET.Element], None]
NoneType = type(None)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sq(x): return x*x

def deg(x): return x*180/pi

def clip(x, l):
    if x < 0: return 0
    if x > l: return l
    return x

def no_comma(s: str) -> str:
    s = s.replace(",", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"

def _find_end(s: str, b: int) -> int:
    n = len(s)
    e = b + 1
    while e < n and s[e] in CHARS:
        e += 1
    return e


def _find_element_name(e: ET.Element, name: str) -> Optional[ET.Element]:
    for child in e:
        try:
            if child.attrib["name"] == name:
                return child
        except:
            pass
    return None


def _find_element_tag(e: ET.Element, tag: str) -> Optional[ET.Element]:
    for child in e:
        if child.tag == tag:
            return child
    return None


def _sensor_fov(sensor: ET.Element, xyz: list[float]):
    x, y, z = xyz

    assert z > 0
    assert x > 0 or y > 0

    if y == 0:
        # -- x & z     fov & fov_axis="x"
        x = x / 2
        l = sqrt(sq(x) + sq(z))
        fov = 2 * deg(asin(x / l))

        efov = ET.Element("float", name="fov", value=fov)
        efov_axis = ET.Element("string", name="fov_axis", value="x")

        sensor.append(efov)
        sensor.append(efov_axis)
    elif x == 0:
        # -- y & z     fov & fov_axis="y"
        y = y / 2
        l = sqrt(sq(y) + sq(z))
        fov = 2 * deg(asin(y / l))

        efov = ET.Element("float", name="fov", value=str(fov))
        efov_axis = ET.Element("string", name="fov_axis", value="y")

        sensor.append(efov)
        sensor.append(efov_axis)
    elif x == y:
        # -- x==y & z     fov
        x = x / 2
        l = sqrt(sq(x) + sq(z))
        fov = 2 * deg(asin(x / l))

        efov = ET.Element("float", name="fov", value=str(fov))

        sensor.append(efov)
    else:
        # -- x!=y & z     fov & fov_axis="diagonal"
        x = x / 2
        y = y / 2
        d = sqrt(sq(x) + sq(y))
        l = sqrt(sq(d) + sq(z))
        fov = 2 * deg(asin(d / l))

        efov = ET.Element("float", name="fov", value=str(fov))
        efov_axis = ET.Element("string", name="fov_axis", value="diagonal")

        sensor.append(efov)
        sensor.append(efov_axis)
    pass
# end

def _is_absolute_path(path:str) -> bool:
    path = path.replace("\\", "/")
    # (windows/linux) \..., /...
    if path.startswith("/"):
        return True
    # (windows) d:...
    if len(path) > 2 and path[1] == ":":
        return True
    return False

# ---------------------------------------------------------------------------
# SceneLoader
# ---------------------------------------------------------------------------

def _float_eval(s: str) -> float:
    if s.startswith("@"):
        return int(s[1:], 10)/255.
    elif s.startswith("#"):
        return int(s[1:], 16)/255.
    else:
        try:
            return float(eval(s))
        except SyntaxError:
            raise SyntaxError(f"syntax error evaluating '{s}' for float")

def _int_eval(s: str) -> int:
    try:
        return int(eval(s))
    except SyntaxError:
        raise SyntaxError(f"syntax error evaluating '{s}' for int")


def _bool_eval(s: str) -> bool:
    try:
        return bool(eval(s))
    except SyntaxError:
        raise SyntaxError(f"syntax error evaluating '{s}' for bool")


def _film_size(sensor: ET.Element, xyz: list[float]):
    x, y, z = xyz
    if x == 0 or y == 0:
        return

    film = _find_element_tag(sensor, "film")
    assert film is not None

    ewidth  = _find_element_name(film, "width")
    eheight = _find_element_name(film, "height")
    if ewidth is not None and eheight is not None:
        return

    if eheight is None:
        width = int(ewidth.attrib["value"])
        height = int(width*y/x)
        eheight = ET.Element("integer", name="height", value=str(height))
        film.append(eheight)
    elif ewidth is None:
        height = int(eheight.attrib["value"])
        width = int(height*x/y)
        ewidth = ET.Element("integer", name="width", value=str(width))
        film.append(ewidth)
    pass
# end


# ---------------------------------------------------------------------------
# SceneLoader
# ---------------------------------------------------------------------------

class SceneLoader:

    def __init__(self):
        self.TAG_IDS = {}
        self.REF_ELEMENTS = {}
        self.DEFAULTS = {}
        self.SCENE_PATH: Path = Path()

        self.shape_depth = 0
        self.parsed = False
        self.scene_dict = {}

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_default_value(self, name: str) -> Union[bool, int, float, str]:
        assert isinstance(name, str)
        assert self.parsed, "Scene not parsed"
        assert name in self.DEFAULTS

        return self.DEFAULTS[name]

    def get_shape(self, id: str) -> dict:
        assert isinstance(id, str)
        assert self.parsed, "Scene not parsed"
        assert id in self.scene_dict

        return self.scene_dict[id]

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def parse_scene(self, scene_path: str|Path, **kwargs) -> "Self":
        """
        Parse the 3D scene defined in the XML file
        :param scene_path: XML file to analyze
        :param kwargs: default values to use durng the parsing.
               They override the default values defined in the file
        """
        self.parsed = False
        if isinstance(scene_path, str):
            scene_path = Path(scene_path)

        assert scene_path.name.endswith(".xml")

        self.SCENE_PATH: Path = scene_path
        self.REF_ELEMENTS.update(kwargs)
        self.shape_depth = 0

        tree = ET.parse(scene_path)
        root = tree.getroot()

        # root = self._resolve_perspective_sensor(root)

        scene_dict = dict()
        self._parse_tag(scene_dict, root)
        self.scene_dict = scene_dict
        self.parsed = True
        return self
    # end

    def get(self) -> dict:
        """Retrieve the 'scene_dict' representing the parsed scene"""
        assert self.parsed, "Scene not parsed"
        return self.scene_dict

    def load_scene(self, scene_path: str|Path, **kwargs) -> dict:
        """
        (Compatibility) Parse the scene and return the generated 'scene_dict'
        :param scene_path: XML file to analyze
        :param kwargs: default values to use durng the parsing.
               They override the default values defined in the file
        :return: the generated 'scene_dict'
        """
        self.parse_scene(scene_path, **kwargs)
        return self.get()

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------

    # def _resolve_default(self, value: str) -> str:
    #     if not value.startswith("$"):
    #         return value
    #
    #     name = value[1:]
    #     # handle "$var1 $var2 ..."
    #     if "$" in name:
    #         return value
    #     # if name not in REF_ELEMENTS:
    #     #     return value
    #
    #     assert name in self.REF_ELEMENTS, f"default value {value} not defined"
    #     return self.REF_ELEMENTS[name]

    # def _resolve_perspective_sensor_root(self, root: ET.Element):
    #     for sensor in root.findall("sensor"):
    #         self._resolve_perspective_sensor(sensor)

    def _resolve_perspective_sensor(self, sensor: ET.Element):
        # instead than fov, to use:
        #
        #   <vector value=""/>
        #   <vector x="" y="" z=""/>
        #
        # possible configurations
        #
        #   x & z       fov & fov_axis="x"
        #   y & z       fov & fov_axis="y"
        #   x,y,z       fov & fov_axis="diagonal"
        #   x==y & z    fov (only)
        #
        #   to force "diagonal" it is enough to have y = (x+eps)
        #
        # if it is specified x & y, it is possible to
        if not sensor.get("type").startswith("perspective"):
            return

        vector = _find_element_tag(sensor, "vector")
        if vector is None:
            return

        xyz = self._parse_array(vector)

        sensor.remove(vector)

        _sensor_fov(sensor, xyz)
        _film_size(sensor, xyz)
        pass
    # end

    def _resolve_default(self, value: str) -> str:
        if not isinstance(value, str):
            return str(value)

        # if "$" not in value:
        #     return value
        while "$" in value:
            b = value.find("$")
            e = _find_end(value, b)
            name = value[b+1:e]
            assert name in self.REF_ELEMENTS, f"default value {value} not defined"
            value = value.replace("$" + name, str(self.REF_ELEMENTS[name]))
        return value
    # end

    def _float(self, x):
        x = self._resolve_default(x)
        return _float_eval(x)

    def _int(self, x) -> int:
        x = self._resolve_default(x)
        return _int_eval(x)

    def _str(self, x) -> str:
        x = self._resolve_default(x)
        return str(x)

    def _bool(self, x) -> bool:
        x = self._resolve_default(x)
        if x in [0, "false", "False", "no"]:
            return False
        if x in [1, "true", "True", "yes"]:
            return True
        assert x in [0, 1, "false", "False", "no", "true", "True", "yes"], f"Boolean value {x} not defined"
        return _bool_eval(x)

    # ---------------------------------------------------------------------------

    def _register_ref(self, xml: ET.Element, data):
        if "id" not in xml.attrib:
            return

        id = xml.attrib["id"]
        self.REF_ELEMENTS[id] = data

    def _get_id(self, xml: ET.Element) -> str:
        if "id" in xml.attrib:
            return xml.attrib["id"]
        if "name" in xml.attrib:
            return xml.attrib["name"]

        tag = xml.tag

        if tag not in self.TAG_IDS:
            self.TAG_IDS[tag] = 1
        id = self.TAG_IDS[tag]
        self.TAG_IDS[tag] += 1
        return f"{tag}@{id}"

    def _get_name(self, xml: ET.Element) -> str:
        if "name" in xml.attrib:
            return xml.attrib["name"]

        tag = xml.tag

        if tag not in self.TAG_IDS:
            self.TAG_IDS[tag] = 1
        id = self.TAG_IDS[tag]
        self.TAG_IDS[tag] += 1
        return f"{tag}@{id}"

    def _get_type(self, xml: ET.Element) -> str:
        assert "type" in xml.attrib
        stype = xml.attrib["type"]
        return self._resolve_default(stype)

    # ---------------------------------------------------------------------------

    def _parse_rgb_color(self, xml: ET.Element) -> list[float]:
        # <tag value="v"/>
        # <tag value="v1, v2, v3"/>
        # <tag value="v1  v2  v3"/>
        # "r g b"
        # "@r @g @b"
        # "#r #g #b"
        # "#rrggbb"
        value = xml.attrib["value"]
        value = no_comma(value)
        parts = value.split(" ")
        assert len(parts) in [1,3]
        if len(parts) == 1 and value.startswith("#"):
            assert value.startswith("#")
            # "#rrggbb"
            rgb = int(value[1:], 16)
            r = (rgb % 256)/255.
            rgb >>= 8
            g = (rgb % 256) / 255.
            rgb >>= 8
            b = (rgb % 256) / 255.
        elif len(parts) == 1:
            c = _float_eval(parts[0])
            r = g = b = c
        else:
            # "@127" "#7F" "0.5"
            r = _float_eval(parts[0])
            g = _float_eval(parts[1])
            b = _float_eval(parts[2])
        return [r, g, b]

    def _parse_array(self, xml: ET.Element) -> list[float]:
        assert isinstance(xml, ET.Element)
        # <tag value="v"/>
        # <tag value="v1, v2, v3"/>
        # <tag value="v1  v2  v3"/>
        # <tag x="..", y="..", z=".."/>
        # ...
        if "x" in xml.attrib or "y" in xml.attrib or "z" in xml.attrib:
            return self._parse_array_xyz(xml)
        else:
            return self._parse_array_value(xml.attrib["value"])

    def _parse_array_xyz(self, xml: ET.Element) -> list[float]:
        x = 0
        y = 0
        z = 0
        if "x" in xml.attrib:
            xvalue = xml.attrib["x"]
            x = self._float(xvalue)
        if "y" in xml.attrib:
            yvalue = xml.attrib["y"]
            y = self._float(yvalue)
        if "z" in xml.attrib:
            zvalue = xml.attrib["z"]
            z = self._float(zvalue)
        return [x, y, z]

    def _parse_array_value(self, value: str | float | list[float]) -> list[float]:
        value = self._resolve_default(value)
        if isinstance(value, (float, list)):
            return value

        value = value.strip()
        if "," in value:
            parts = value.split(",")
            values = list(map(lambda s: self._float(s.strip()), parts))
            return values
        elif " " in value:
            parts = value.split(" ")
            parts = [p for p in parts if len(p) > 0]
            values = list(map(lambda s: self._float(s.strip()), parts))
            return values
        else:
            value = self._float(value.strip())
            return [value, value, value]

    def _parse_matrix_value(self, value: str) -> list[list[float]]:
        data = self._parse_array_value(value)
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

    # def _parse_body(self, data: dict, xml: ET.Element):
    #     for child in xml:
    #         self._parse_tag(data, child)

    def _parse_tag(self, data: dict, xml: ET.Element):
        assert isinstance(data, dict)
        assert isinstance(xml, ET.Element)
        assert xml.tag in self.TAG_PARSERS.keys(), f"Tag {xml.tag} unsupported"

        parse: TAG_FUNCTION = cast(TAG_FUNCTION, self.TAG_PARSERS[xml.tag])
        parse(self, data, xml)

    def _parse_scene(self, data: dict, xml: ET.Element):
        assert xml.tag == "scene"
        #   <scene>
        #   </scene>
        data["type"] = "scene"
        self._parse_children(data, xml)

    def _parse_body(self, data: dict, xml: ET.Element):
        # just an alias
        self._parse_children(data, xml)

    def _parse_children(self, data: dict, xml: ET.Element):
        nelts = len(xml)
        for i in range(nelts):
            child = xml[i]
            self._parse_tag(data, child)

    def _parse_shape(self, data: dict, xml: ET.Element):
        assert xml.tag == "shape"
        #   <shape type="">
        #       ...
        #   </shape>
        id = self._get_id(xml)
        type = self._get_type(xml)
        self.shape_depth += 1

        if type in ["obj", "ply"]:
            self._resolve_filename(data, xml)

        #
        shape = dict(type=type, id=id)
        self._parse_children(shape, xml)
        self.shape_depth -= 1
        #
        if self.shape_depth == 0:
            self._register_ref(xml, shape)
        data[id] = shape
        pass

    def _parse_integrator(self, data: dict, xml: ET.Element):
        assert xml.tag == "integrator"
        #   <integrator type="">
        #       <integer name="max_depth" value="5"/>
        #   </integrator>
        id = self._get_id(xml)
        type = self._get_type(xml)
        #
        integrator = dict(type=type, id=id)
        self._parse_children(integrator, xml)
        #
        self._register_ref(xml, integrator)
        data[id] = integrator
        pass

    def _parse_sensor(self, data: dict, xml: ET.Element):
        assert xml.tag == "sensor"
        #   <sensor type="">
        #       ...
        #       <sampler type="..."> ... </sampler>
        #       <film type="..."> ... <film>
        #   </sensor>
        id = self._get_id(xml)
        type = self._get_type(xml)
        #
        sensor = dict(type=type, id=id)

        self._resolve_perspective_sensor(xml)
        self._parse_children(sensor, xml)
        #
        self._register_ref(xml, sensor)
        data[id] = sensor
        pass

    def _parse_sampler(self, data: dict, xml: ET.Element):
        assert xml.tag == "sampler"
        #   <sampler type="">
        #       ...
        #   </sampler>
        id = self._get_id(xml)
        type = self._get_type(xml)
        #
        sampler = dict(type=type, id=id)
        self._parse_children(sampler, xml)
        #
        self._register_ref(xml, sampler)
        data[id] = sampler
        pass

    def _parse_film(self, data: dict, xml: ET.Element):
        assert xml.tag == "film"
        #   <film type="">
        #       ...
        #   </film>
        id = self._get_id(xml)
        type = self._get_type(xml)
        #
        film = dict(type=type, id=id)
        self._parse_children(film, xml)
        #
        self._register_ref(xml, film)
        data[id] = film
        pass

    def _parse_emitter(self, data: dict, xml: ET.Element):
        assert xml.tag == "emitter"
        #   <emitter type="">
        #       ...
        #   </emitter>
        id = self._get_id(xml)
        type = self._get_type(xml)
        #
        emitter = dict(type=type, id=id)
        self._parse_children(emitter, xml)
        #
        self._register_ref(xml, emitter)
        data[id] = emitter
        pass

    def _parse_bsdf(self, data: dict, xml: ET.Element):
        assert xml.tag == "bsdf"
        #   <bsdf type="">
        #       ...
        #   </bsdf>
        id = self._get_id(xml)
        type = self._get_type(xml)
        #
        bsdf = dict(type=type, id=id)
        self._parse_children(bsdf, xml)
        #
        self._register_ref(xml, bsdf)
        data[id] = bsdf
        pass

    def _parse_rfilter(self, data: dict, xml: ET.Element):
        assert xml.tag == "rfilter"
        #   <rfilter type="">
        #       ...
        #   </rfilter>
        id = self._get_id(xml)
        type = self._get_type(xml)
        #
        rfilter = dict(type=type, id=id)
        self._parse_children(rfilter, xml)
        #
        self._register_ref(xml, rfilter)
        data[id] = rfilter
        pass

    def _parse_ref(self, data: dict, xml: ET.Element):
        assert xml.tag == "ref"
        # <ref id="..." name="..." />
        id = xml.attrib["id"]
        name = xml.attrib["name"]
        id = self._resolve_default(id)
        # assert id in REF_ELEMENTS, f"Reference {id} not found"
        # return REF_ELEMENTS[id]
        data[name] = dict(type="ref", id=id)
        pass

    def _parse_transform(self, data: dict, xml: ET.Element):
        assert xml.tag == "transform"
        #   <transform name="">
        #
        #   <transform>
        name = xml.attrib["name"]
        #
        t = mi.ScalarTransform4f()
        nt = len(xml)
        for i in range(nt):
            t = self._apply_transform(t, xml[i])

        data[name] = t
        pass

    # ---------------------------------------------------------------------------

    def _parse_spectrum(self, data: dict, xml: ET.Element):
        assert xml.tag == "spectrum"
        raise UnsupportedOperation()

    def _parse_texture(self, data: dict, xml: ET.Element):
        assert xml.tag == "texture"
        # <texture name="" type="...">
        # </texture>
        name = xml.attrib["name"]
        type = xml.attrib["type"]

        texture = dict(type=type)
        self._parse_children(texture, xml)
        #
        data[name] = texture
        # raise UnsupportedOperation()
        pass

    def _parse_include(self, data: dict, xml: ET.Element):
        assert xml.tag == "include"
        raise UnsupportedOperation()

    def _parse_alias(self, data: dict, xml: ET.Element):
        assert xml.tag == "alias"
        # <alias id="top-camera" as="default-camera"/>
        # oid = xml.attrib["id"]
        # asid = xml.attrib["as"]
        # obj = data[oid]
        # obj["id"] = asid
        # data[asid] = obj
        # raise UnsupportedOperation()
        pass

    def _parse_path(self, data: dict, xml: ET.Element):
        assert xml.tag == "path"
        raise UnsupportedOperation()

    # ---------------------------------------------------------------------------

    def _parse_boolean(self, data: dict, xml: ET.Element):
        assert xml.tag == "boolean"
        name = xml.attrib["name"]
        value = xml.attrib["value"]
        data[name] = self._bool(value)
        pass

    def _parse_integer(self, data: dict, xml: ET.Element):
        assert xml.tag == "integer"
        name = xml.attrib["name"]
        value = xml.attrib["value"]
        data[name] = self._int(value)
        pass

    def _parse_float(self, data: dict, xml: ET.Element):
        assert xml.tag == "float"
        name = xml.attrib["name"]
        value = xml.attrib["value"]
        data[name] = self._float(value)
        pass

    def _parse_string(self, data: dict, xml: ET.Element):
        assert xml.tag == "string"
        name = xml.attrib["name"]
        value = xml.attrib["value"]

        if name == "filename" and not _is_absolute_path(value):
            parent = str(self.SCENE_PATH.parent).replace("\\","/")
            if len(parent) > 0:
                value = f"{parent}/{value}"

        data[name] = self._str(value)
        pass

    def _parse_vector(self, data: dict, xml: ET.Element):
        assert xml.tag == "vector"
        # <vector name="" value=""/>
        # <vector name="" x="", y="", z=""/>
        # name = xml.attrib["name"]
        name = self._get_name(xml)
        point = self._parse_array(xml)

        data[name] = point
        pass

    def _parse_point(self, data: dict, xml: ET.Element):
        assert xml.tag == "point"
        # <point name="" value=""/>
        # <point name="" x="", y="", z=""/>
        name = xml.attrib["name"]
        point = self._parse_array(xml)

        data[name] = point
        pass

    def _parse_rgb(self, data: dict, xml: ET.Element):
        assert xml.tag == "rgb"
        # <rgb name="intensity" value="1"/>
        name = xml.attrib["name"]
        value = self._parse_rgb_color(xml)

        data[name] = dict(
            type="rgb",
            value=value
        )
        pass

    def _parse_default(self, data: dict, xml: ET.Element):
        assert xml.tag == "default"
        # <default name="" value=""/>
        name = xml.attrib["name"]
        value = xml.attrib["value"]

        if "$" in value:
            value = f"({self._resolve_default(value)})"

        if name not in self.REF_ELEMENTS:
            self.REF_ELEMENTS[name] = value
            self.DEFAULTS[name] = value
        # elif value == self.REF_ELEMENTS[name]:
        #     pass
        else:
            print(f"Default {name}: config={value}, set={self.REF_ELEMENTS[name]}")

        # data[name] = dict(type="default", value=value)
        pass

    # ---------------------------------------------------------------------------

    def _parse_medium(self, data: dict, xml: ET.Element):
        assert xml.tag == "medium"
        # <medium type="" id="">
        #   float|string "int_ior"
        #   float|string "ext_ior"
        #   spectrum|texture    "specular_reflectance"
        #   spectrum|texture    "specular_transmittance"
        # </medium>
        id = self._get_id(xml)
        type = xml.attrib["type"]

        medium = dict(type=type, id=id)
        self._parse_children(medium, xml)
        #
        self._register_ref(xml, medium)
        data[id] = medium
        pass

    def _parse_phase(self, data: dict, xml: ET.Element):
        assert xml.tag == "phase"
        # <phase type="" id="">
        # </phase>
        id = self._get_id(xml)
        type = xml.attrib["type"]

        phase = dict(type=type, id=id)
        self._parse_children(phase, xml)
        #
        self._register_ref(xml, phase)
        data[id] = phase
        pass

    def _parse_volume(self, data: dict, xml: ET.Element):
        assert xml.tag == "volume"
        # <volume type="", name="">
        # </volume>
        name = xml.attrib["name"]
        type = xml.attrib["type"]

        volume = dict(type=type)
        self._parse_children(volume, xml)
        #
        data[name] = volume
        pass

    # ---------------------------------------------------------------------------

    def _resolve_filename(self, data: dict, xml: ET.Element):
        type =  xml.attrib["type"]
        xfn = _find_element_name(xml, "filename")
        filename = xfn.attrib["value"]
        if not Path(filename).exists():
            path = Path(__file__).parent / f"{type}/{filename}"
            filename = str(path)
            xfn.attrib["value"] = filename
        pass

    # ---------------------------------------------------------------------------


    TAG_PARSERS: dict[str, Any] = {
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
        "path": _parse_path,

        "medium": _parse_medium,
        "phase": _parse_phase,
        "volume": _parse_volume
    }

    # ---------------------------------------------------------------------------
    # Transformations
    # ---------------------------------------------------------------------------

    def _parse_rotate(self, t, xml: ET.Element):
        assert xml.tag == "rotate"
        # <rotate x="1" angle="5"/>
        # <rotate value="1,2,3" angle="5>
        axis = self._parse_array(xml)
        angle = self._float(xml.attrib["angle"])

        tr = mi.ScalarTransform4f().rotate(axis, angle)
        return tr @ t

    def _parse_scale(self, t, xml: ET.Element):
        assert xml.tag == "scale"
        # <scale value="5"/>
        # <scale value="1,2,3">
        v = self._parse_array(xml)
        ts = mi.ScalarTransform4f().scale(v)
        return ts @ t

    def _parse_translate(self, t, xml: ET.Element):
        assert xml.tag == "translate"
        # <translate value="5"/>
        # <translate value="1,2,3">
        v = self._parse_array(xml)
        tt = mi.ScalarTransform4f().translate(v)
        return tt @ t

    def _parse_matrix(self, t, xml: ET.Element):
        assert xml.tag == "matrix"
        # <matrix value="-1 0 0 0 0 1 0 1 0 0 -1 6.8 0 0 0 1" />
        m = self._parse_matrix_value(xml.attrib["value"])

        tm = mi.ScalarTransform4f(m)
        return tm @ t

    def _parse_lookat(self, t, xml: ET.Element):
        assert xml.tag == "lookat"
        #   <lookat target="..." origin="..." up="..."/>
        target = self._parse_array_value(xml.attrib["target"])
        origin = self._parse_array_value(xml.attrib["origin"])
        up = self._parse_array_value(xml.attrib["up"])

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

    def _apply_transform(self, t, xml: ET.Element):
        assert isinstance(xml, ET.Element)
        assert xml.tag in self.TAG_TRANSFORMS.keys(), f"Transform {xml.tag} unsupported"

        transform = self.TAG_TRANSFORMS[xml.tag]
        return transform(self, t, xml)
    # end
# end


# ---------------------------------------------------------------------------
# load_scene_xml
# ---------------------------------------------------------------------------

def load_scene_xml(scene_xml: str|Path, **kwargs) -> dict:
    assert isinstance(scene_xml, (str, Path))
    # global TAG_IDS, REF_ELEMENTS
    sl = SceneLoader()
    scene_dict = sl.load_scene(scene_xml, **kwargs)
    return scene_dict
# end

load_scene_dict = load_scene_xml


class MitsubaJSONEncoder(json.JSONEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default(self, o):
        if isinstance(o, mi.ScalarAffineTransform4f):
            return list(o.matrix)
        elif isinstance(o, dr.scalar.Array4f):
            return list(o)
        else:
            pass
        return super().default(o)


def save_scene_dict(scene_json: str|Path, scene_dict: dict):
    assert isinstance(scene_json, (str, Path))
    assert isinstance(scene_dict, dict), "Scene must be a dict"

    with open(scene_json, "w") as f:
        json.dump(scene_dict, f, indent=4, cls=MitsubaJSONEncoder)
    pass
# end


def load_scene(scene_xml: str|Path, **kwargs) -> "mi.Scene":
    assert isinstance(scene_xml, (str, Path))
    scene_dict = load_scene_xml(scene_xml, **kwargs)
    return mi.load_dict(scene_dict)
# end


def load_dict(scene_dict: dict) -> "mi.Scene":
    assert isinstance(scene_dict, dict), "Scene must be a dict"
    return mi.load_dict(scene_dict)
# end


# ---------------------------------------------------------------------------
#
# --------------------------------------------------------------------------

def get_by_id(scene: "mi.Scene", obj_id: str) -> mi.Object:
    """
    Retrieve the object with the specified id in Scene object

    :param scene:
    :param obj_id:
    :return:
    """
    assert isinstance(scene, mi.Scene)

    for obj in cast(list[mi.Sensor], scene.sensors()):
        if obj.id() == obj_id:
            return obj

    for obj in cast(list[mi.Shape], scene.shapes()):
        if obj.id() == obj_id:
            return obj

    for obj in cast(list[mi.Emitter], scene.emitters()):
        if obj.id() == obj_id:
            return obj

    return None

# ---------------------------------------------------------------------------
#
# --------------------------------------------------------------------------

class ToWorld:
    """
    World transformation
    It permits composition a transformation one element at time.
    Primitive transformations:
    - translation
    - scale
    - rotation
    - look_at
    - matrix
    - another transformation
    The composed transformation is applied at the left  (newt @ oldt)
    The result is a matrix that can be used to transform points and vectors
    and it is obtained with 'get()'
    """

    def __init__(self, t=None):
        assert t is None or isinstance(t, mi.ScalarTransform4f)
        self.t = mi.ScalarTransform4f() if t is None else t

    def add(self, ta):
        assert isinstance(ta, mi.ScalarTransform4f)
        self.t = ta @ self.t
        return self

    def translate(self, x=0,y=0,z=0,value=None):
        if value is not None:
            x,y,z = value

        t = self.t
        tt = mi.ScalarTransform4f().translate([x,y,z])
        self.t = tt @ t
        return self

    def scale(self, x=1, y=1, z=1, value=None):
        if isinstance(value, (int, float)):
            v = value
            value = [v,v,v]
        if value is not None:
            x,y,z = value

        t = self.t
        ts = mi.ScalarTransform4f().scale([x,y,z])
        self.t = ts @ t
        return self

    def rotate(self, x=0, y=0, z=0, value=None, angle=0):
        if isinstance(value, (int, float)):
            v = value
            value = [v,v,v]
        if value is not None:
            x,y,z = value

        t = self.t
        tr = mi.ScalarTransform4f().rotate([x,y,z], angle)
        self.t = tr @ t
        return self

    def look_at(self, origin=(0,0,1), target=(0,0,0), up=(0,0,1)):

        t = self.t
        tl = mi.ScalarTransform4f().look_at(origin=origin, target=target, up=up)
        self.t = tl @ t
        return self

    def matrix(self, data: list[float]):
        assert len(data) in [9, 16], "Matrix must be [a11, a12, ...]"
        if len(data) == 9:
            m = [
                data[0:3],
                data[3:6],
                data[6:]
            ]
        else:
            m = [
                data[0:4],
                data[4:8],
                data[8:12],
                data[12:]
            ]

        t = self.t
        tm = mi.ScalarTransform4f(m)
        self.t = tm @ t
        return self

    def get(self):
        return self.t
# end


# ---------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def render(scene: object,
           gamma=2.2,
           params: Any = None,
           sensor: int = 0,
           integrator = None,
           seed = 0,
           seed_grad: int = 0,
           spp: int = 0,
           spp_grad: int = 0) -> np.ndarray:
    """
    Call the Mitsuba renderer and apply the Gamma correction
    :param scene:
    :param gamma: Gamma correction factor
    :param params:
    :param sensor:
    :param integrator:
    :param seed:
    :param seed_grad:
    :param spp:
    :param spp_grad:
    :return:
    """
    rimage = mi.render(
        scene=scene,
        params=params,
        sensor=sensor,
        integrator=integrator,
        seed=seed,
        seed_grad=seed_grad,
        spp=spp,
        spp_grad=spp_grad,
    )

    image: np.ndarray = np.array(rimage)
    image = (image ** (1. / gamma)).clip(0, 1)

    return image
# end

# ---------------------------------------------------------------------------
# instance
# WARNING: 'instance' of a 'shapegroup' it slower to clone the object
# ---------------------------------------------------------------------------

def instance(scene_dict: dict, *, id: str, ref: str, to_world, in_scene=True, **kwargs) -> dict:
    """
    Create an instance of a shapegroup and add it in the scene
    :param scene_dict: scene
    :param id: id of the instance. It must be not None and unique
    :param ref: object of type 'shapegroup' to instantiate
    :param to_world: to world transformation
    :param in_scene: if to insert the object in the scene
    :return: instance object
    """
    if isinstance(to_world, ToWorld):
        to_world = to_world.get()

    assert isinstance(scene_dict, dict), f"Scene must be a dict"
    assert isinstance(id, str) and id not in scene_dict, f"Object id {id} already used"
    assert isinstance(ref, str) and ref in scene_dict, f"Referenced object {ref} not in scene"
    assert isinstance(to_world, mi.ScalarTransform4f)

    instance = {
        "id": id,
        "type":"instance",
        "shape": {"type": "ref", "id": ref},
        "to_world": to_world
    }

    if in_scene:
        scene_dict[id] = instance

    return instance
# end


# ---------------------------------------------------------------------------
# clone
# support to clone a "shapegroup" and to replace the default values
# --------------------------------------------------------------------------

def _clone_shape(scene_dict: dict, id: str, shape: dict, to_world, in_scene=True,
                      sgt:dict[str, "mi.ScalarTransform4f"]={}) -> list[dict]:
    if shape["type"] == "shapegroup":
        return _clone_shapegroup(scene_dict, id, shape, to_world, in_scene, sgt)
    else:
        return _clone_primitive(scene_dict, id, shape, to_world, in_scene, sgt)


def _clone_primitive(scene_dict: dict, id: str, shape: dict, to_world, in_scene=True,
                      sgt:dict[str, "mi.ScalarTransform4f"]={}) -> list[dict]:

    clone = {} | shape

    clone["id"] = id
    clone["to_world"] = ToWorld(clone.get("to_world", mi.ScalarTransform4f())).add(to_world).get()

    if in_scene:
        scene_dict[id] = clone

    return [clone]


def _clone_shapegroup(scene_dict: dict, id: str, shapegroup: dict, to_world, in_scene=True,
                      sgt:dict[str, "mi.ScalarTransform4f"]={}) -> list[dict]:
    clones = []

    for sid in shapegroup:
        shape = shapegroup[sid]
        if not isinstance(shape, dict): continue
        if not "type" in shape: continue

        if sid in sgt:
            st = shape.get("to_world", mi.ScalarTransform4f())
            shape["to_world"] = sgt[sid] @ st
            pass

        cid = f"{id}_{sid}"
        clone = _clone_shape(scene_dict, cid, shape, to_world, in_scene)
        clones.append(clone)

    return clones


def clone(scene_dict: dict, id: str, ref: str, to_world, in_scene=True,
          sgt:dict[str, "mi.ScalarTransform4f"]={},
          template=False) -> list[dict]:
    """
    Clone an object in the scene
    :param scene_dict: scene dictionary
    :param sid: shape id of the instance. It must be not None and unique
    :param ref: object to clone
    :param to_world: world transformation
    :param in_scene: if to insert the object in the scene
    :param sgt: shapegroup transform: dictionary "id_path": mi.Transform4f
    :return: cloned object
    """
    if isinstance(to_world, ToWorld):
        to_world = to_world.get()

    assert isinstance(scene_dict, dict), f"Scene must be a dict"
    assert isinstance(id, str) and id not in scene_dict, f"Object id {id} already used"
    assert isinstance(ref, str) and ref in scene_dict, f"Referenced object {ref} not in scene"
    assert isinstance(to_world, mi.ScalarTransform4f)

    shape = scene_dict[ref]
    clone: list[dict] = _clone_shape(scene_dict, id, shape, to_world, in_scene, sgt)

    # if shape["type"] == "shapegroup":
    #     clone = _clone_shapegroup(scene_dict, id, shape, to_world, in_scene, sgt)
    # else:
    #     clone = _clone_primitive(scene_dict, id, shape, to_world, in_scene)
    if template:
        for c in clone:
            c["template"] = ref

    return clone
# end


# ---------------------------------------------------------------------------
# bounding_box
# ---------------------------------------------------------------------------
# Compute the bounding_box of a shape

SHAPE_TYPES = {
    "obj", "ply", "serialized",
    "cube", "sphere", "rectangle", "disk", "cylinder",
    "bsplinecurve", "linearcurve", "sdfgrid",
    "shapegroup", "instance",
    "ellipsoids", "ellipsoidsmesh",

}

def _load_ply_points(filename: str) -> "mi.Point3f":
    raise NotImplementedError()


def _load_obj_points(filename: str) -> "mi.Point3f":
    # it tries to load the filename base on the directory containing the scene XML.
    # if not available, it tries to load the file from [MITSUBAX]/<type>/<filename>
    if not Path(filename).exists():
        path = Path(__file__).parent / f"obj/{filename}"
        filename = str(path)
        pass

    coords = []
    with open(filename, "r") as f:
        for line in f:
            if not line.startswith("v "): continue
            coords.append([float(x) for x in line.split()[1:]])
    pass
    return mi.Point3f(np.array(coords).T)

def _pts_instance(scene_dict: dict, shape: dict):
    ref_shape = shape["shape"]["id"]
    points = _pts_shape(scene_dict, ref_shape)
    if "to_world" in shape:
        to_world: mi.Transform4f = mi.Transform4f(shape["to_world"])
        points = to_world @ points
    return points

def _pts_obj(scene_dict: dict, shape: dict):
    filename = shape["filename"]
    points: mi.Point3f = _load_obj_points(filename)
    if "to_world" in shape:
        to_world: mi.Transform4f = mi.Transform4f(shape["to_world"])
        points = to_world @ points
    return points

def _pts_ply(scene_dict: dict, shape: dict):
    raise NotADirectoryError()

def _pts_serialized(scene_dict: dict, shape: dict):
    raise NotADirectoryError()

def _pts_cube(scene_dict: dict, shape: dict):
    vertices = np.array([
        [-1,-1,-1],
        [ 1,-1,-1],
        [ 1, 1,-1],
        [-1, 1,-1],

        [-1,-1, 1],
        [1, -1, 1],
        [1,  1, 1],
        [-1, 1, 1],
    ], dtype=float)
    points = mi.Point3f(vertices.T)
    if "to_world" in shape:
        to_world: mi.Transform4f = mi.Transform4f(shape["to_world"])
        points = to_world @ points
    return points

def _pts_sphere(scene_dict: dict, shape: dict):
    return _pts_cube(scene_dict, shape)

def _pts_rectangle(scene_dict: dict, shape: dict):
    vertices = np.array([
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
    ], dtype=float)
    points = mi.Point3f(vertices.T)
    if "to_world" in shape:
        to_world: mi.Transform4f = mi.Transform4f(shape["to_world"])
        points = to_world @ points
    return points

def _pts_disk(scene_dict: dict, shape: dict):
    return _pts_rectangle(scene_dict, shape)

def _pts_cylinder(scene_dict: dict, shape: dict):
    raise NotADirectoryError()

def _pts_bsplinecurve(scene_dict: dict, shape: dict):
    raise NotADirectoryError()

def _pts_linearcurve(scene_dict: dict, shape: dict):
    raise NotADirectoryError()

def _pts_sdfgrid(scene_dict: dict, shape: dict):
    raise NotADirectoryError()

def _pts_shapegroup(scene_dict: dict, shape: dict):
    pts_list = []
    for key in shape:
        value = shape[key]
        if not isinstance(value, dict): continue
        if not "type" in value: continue
        pts = _pts_shape(scene_dict, value)
        pts_list.append(pts)
    # end
    points = points_concat(pts_list)
    if "to_world" in shape:
        to_world: mi.Transform4f =  mi.Transform4f(shape["to_world"])
        points = to_world @ points
    return points

def _pts_ellipsoids(scene_dict: dict, shape: dict):
    raise NotADirectoryError()

def _pts_ellipsoidsmesh(scene_dict: dict, shape: dict):
    raise NotADirectoryError()

PTS_DICT: dict[str, Callable] = {
    "instance": _pts_instance,
    "obj": _pts_obj,
    "ply": _pts_ply,
    "serialized": _pts_serialized,
    "cube": _pts_cube,
    "sphere": _pts_sphere,
    "rectangle": _pts_rectangle,
    "disk": _pts_disk,
    "cylinder": _pts_cylinder,
    "bsplinecurve": _pts_bsplinecurve,
    "linearcurve": _pts_linearcurve,
    "sdfgrid": _pts_sdfgrid,
    "shapegroup": _pts_shapegroup,
    "ellipsoids": _pts_ellipsoids,
    "ellipsoidsmesh": _pts_ellipsoidsmesh,
}

# ---------------------------------------------------------------------------

def _pts_shape(scene_dict: dict, sid: str|dict) -> "mi.Point3f":
    assert isinstance(sid, (str, dict))

    if isinstance(sid, dict):
        shape = sid
        type = shape["type"]
        _pts_fun = PTS_DICT[type]
        points: mi.Point3f = _pts_fun(scene_dict, shape)
    elif sid in scene_dict:
        shape = scene_dict[sid]
        type = shape["type"]
        _pts_fun = PTS_DICT[type]
        points: mi.Point3f = _pts_fun(scene_dict, shape)
    else:
        pts_list = []
        prefix = f"{sid}_"
        for sid in scene_dict:
            if sid.startswith(prefix) and is_shape(scene_dict[sid]):
                shape = scene_dict[sid]
                type = shape["type"]
                _pts_fun = PTS_DICT[type]
                pts: mi.Point3f = _pts_fun(scene_dict, shape)
                pts_list.append(pts)
        points = points_concat(pts_list)
    # end
    return points


def points_concat(points: list["mi.Point3f"]) -> "mi.Point3f":
    """
    Concatenate the list of Point3f in a single Point3f
    :param points: points to concatenate
    :return: single point
    """
    assert isinstance(points, list)

    # x = []
    # y = []
    # z = []
    # for point in points:
    #     assert isinstance(point, mi.Point3f)
    #
    #     x += list(point.x)
    #     y += list(point.y)
    #     z += list(point.z)
    # # end
    # return mi.Point3f(x,y,z)

    px = [p.x for p in points]
    py = [p.y for p in points]
    pz = [p.z for p in points]

    x = dr.concat(px)
    y = dr.concat(py)
    z = dr.concat(pz)

    return mi.Point3f(x, y, z)



def is_shape(shape) -> bool:
    if not isinstance(shape, dict): return False
    if "type" not in shape: return False
    type = shape["type"]
    if type not in SHAPE_TYPES: return False
    return True


def scene_points(scene_dict: dict, id_list: Union[None,str,list[str]]=None)-> "mi.Point3f":
    """
    Collect the points used to draw the shapes in the list
    :param scene_dict: scene to analyze
    :param id_list: list of shapes' id to analyze. If None or [], analyze all shapes in scene
    :return: a Point3f with the list of points
    """
    assert isinstance(scene_dict, dict), "Scene must be a dict"
    if id_list is None: id_list = []
    if isinstance(id_list, str): id_list=[id_list]
    assert isinstance(id_list, list), "id_list must be a list of shape ids"

    # if [] select all shapes in scene
    if len(id_list) == 0:
        for id in scene_dict:
            if is_shape(scene_dict[id]):
                id_list.append(id)

    pts_list: list[mi.Point3f] = []
    for id in id_list:
        pts = _pts_shape(scene_dict, id)
        pts_list.append(pts)
    points = points_concat(pts_list)
    return points
# end


def bounding_box(scene_dict: dict, id_or_shape: str|list[str]|dict) -> "mi.BoundingBox3f":
    """
    Compute the boundingbox of the shape in the scene

    :param scene_dict: scene
    :param id: id or shapeto analyze
    :return:
    """
    # Note: IF id is not in scene, it is possible there are multiple ids starting with 'id.'
    assert isinstance(scene_dict, dict), "Scene must be a dict"
    assert isinstance(id_or_shape, (str, list, dict)), "Sid must be a string or a dictionary"

    if isinstance(id_or_shape, dict):
        points = _pts_shape(scene_dict, id_or_shape)
    else:
        points = scene_points(scene_dict, id_or_shape)

    # if id_or_shape in scene_dict:
    #     points: mi.Point3f = _pts_shape(scene_dict, id_or_shape)
    # elif isinstance(id_or_shape, str):
    #     prefix = f"{id_or_shape}_"
    #     pts_list: list[mi.Point3f] = []
    #     for sid in scene_dict:
    #         if sid.startswith(prefix):
    #             pts = _pts_shape(scene_dict, sid)
    #             pts_list.append(pts)
    #
    #     assert len(pts_list) > 0
    #     points: mi.Point3f = points_concat(pts_list)
    # else:
    #     points: mi.Point3f = _pts_shape(scene_dict, id_or_shape)

    x = list(points.x)
    y = list(points.y)
    z = list(points.z)

    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    minz = min(z)
    maxz = max(z)

    minp = mi.Point3f(minx,miny,minz)
    maxp = mi.Point3f(maxx, maxy,maxz)

    return mi.BoundingBox3f(minp, maxp)


def bounding_cube(bbox: "mi.BoundingBox3f", z: Union[None, bool, int, float]=None) -> "mi.Point3f":
    """
    Convert a bounding box, containing only the points composed by the min and max coordinates
    into the list of coordinatesof of the correspondent parallelepiped

    :param bbox: bounding box
    :param z: if it is true, instead than a cube, it returns a rectangle located at (zmin+zmax)/2
    :return: parallelepiped coordinates
    """
    assert isinstance(bbox, mi.BoundingBox3f)
    assert isinstance(z, (NoneType, bool, int, float))

    minx = bbox.min.x[0]
    maxx = bbox.max.x[0]
    miny = bbox.min.y[0]
    maxy = bbox.max.y[0]
    minz = bbox.min.z[0]
    maxz = bbox.max.z[0]

    if z is None:
        coords = [
            [minx, miny, minz],
            [maxx, miny, minz],
            [maxx, maxy, minz],
            [minx, maxy, minz],

            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, maxy, maxz],
            [minx, maxy, maxz],
        ]
        return mi.Point3f(np.array(coords).T)

    if z == False: z = 0
    if z == True:  z = 1
    assert 0 <= z <= 1, f"Invalid 'z' value ({z}). 'z' must be in range [0,1]"

    meanz = minz + z*(maxz-minz)
    coords = [
        [minx, miny, meanz],
        [maxx, miny, meanz],
        [maxx, maxy, meanz],
        [minx, maxy, meanz],
    ]
    return mi.Point3f(np.array(coords).T)


FORMAT_TYPE = Literal["xxyy", "xyxy", "xywh", "cxcywh", "yolo", "voc"]

def sensor_bbox(sensor: "mi.Sensor", scene_dict: dict, id: str, format: FORMAT_TYPE="xywh") \
    -> tuple[int,int,int,int]:
    """

    :param sensor:
    :param scene_dict:
    :param id:
    :param format:
                xyxy: VOC, tocrhvision
                xywh:
                cxcywh: YOLO
    :return:
    """
    assert isinstance(sensor, mi.Sensor)
    assert isinstance(scene_dict, dict)
    assert isinstance(id, str)
    # assert isinstance(format, FORMAT_TYPE)
    assert format in ["xxyy", "xyxy", "xywh", "cxcywh", "yolo", "voc"]

    w, h = sensor.film().size()

    # 1) collect the 3D points representing the shape
    points = _pts_shape(scene_dict, id)
    # 2) project the points on the sensor
    spoints, _ = sensor.project_point(points, active=True)
    spoints = np.array(spoints)
    xmin = clip(spoints[0].min(), w)
    ymin = clip(spoints[1].min(), h)
    xmax = clip(spoints[0].max(), w)
    ymax = clip(spoints[1].max(), h)

    if xmin == 0: xmin = xmax
    if ymin == 0: ymin = ymax

    # dx = xmax - xmin
    # dy = ymax - ymin
    #
    # if dx > 100:
    #     print("x", dx, xmin, xmax)
    # if dy > 100:
    #     print("y", dy, ymin, ymax)

    if format in ["xxyy"]:
        return (xmin), (xmax), (ymin), (ymax)
    elif format in ["xyxy", "voc"]:        # VOC, torchvision
        return (xmin), (ymin), (xmax), (ymax)
    elif format in ["xywh"]:
        return (xmin), (ymin), (xmax-xmin), (ymax-ymin)
    elif format in ["cxcywh", "yolo"]:    # YOLO
        return ((xmin+xmax)/2), ((ymin+ymax)/2), (xmax-xmin), (ymax-ymin)
    else:
        raise ValueError(f"Unknown format '{format}'")
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
