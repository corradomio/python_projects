from typing import *
from types import *
from collections import *
from stdlib.is_instance import is_instance, InRange


assert is_instance(3, InRange[0,2])