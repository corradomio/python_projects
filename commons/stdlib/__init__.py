__version__ = "1.1.15"

from .qname import module_path, import_from, qualified_name, qualified_type, name_of, ns_of
from .convert import tobool, as_dict, as_list, as_tuple, tofloat, to_bool, to_float
from .convert import NoneType, RangeType, CollectionType, FunctionType
from .collections import list_map, lrange, mul_, sum_, prod_, argsort
from .kwargs import kwexclude, kwparams, kwselect, kwval, kwmerge
from .bag import bag
from .dict import dict
from .dictx import dict_select, dict_exclude, dict_get
from .is_instance import is_instance, IS_INSTANCE_OF, IsInstance
from .language import method_of
