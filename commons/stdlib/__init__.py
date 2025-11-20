__version__ = "1.1.15"

from .collections import list_map, lrange, mul_, sum_, prod_, argsort
from .convert import tobool, as_dict, as_list, as_tuple, tofloat, to_bool, to_float, as_str
from .dictx import dict_select, dict_exclude, dict_get
from .is_instance import is_instance, IS_INSTANCE_OF, IsInstance
from .kwargs import kwexclude, kwparams, kwselect, kwval, kwmerge
from .language import method_of
from .qname import module_path, qualified_name, qualified_type, name_of, ns_of, class_of
from .qname import import_from, create_from, create_from_collection
from .types import NoneType, RangeType, CollectionType, LambdaType