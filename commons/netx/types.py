from typing import Union

NODE_TYPE = Union[int, str]
EDGE_TYPE = tuple[NODE_TYPE, NODE_TYPE] | tuple[NODE_TYPE, NODE_TYPE, dict]
