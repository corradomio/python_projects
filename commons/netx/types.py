from typing import Union, Tuple, List

NODE_TYPE = Union[int, str]
EDGE_TYPE = Union[Tuple[Union[NODE_TYPE,dict], ...], List[Union[NODE_TYPE,dict]]]
