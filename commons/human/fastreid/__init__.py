# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

@author:    corradomio
@contact: corrado.mio@gmail.com, corrado.mio@ku.ac.ae
"""

# Patches:
# 1) import "Mapping" from "collections.abc" and NOT from "collections"
# 2) removed "from forch._six import string_classes"
#    and replaced with "string_classes = (str,)"
#

from ._fastreid import FASTREID_MODEL_NAMES, FASTREID_WEIGHTS_ROOT
from ._fastreid import FastReID


__version__ = "1.4.1"
