from typing import Union

import numpy as np
from numbers import Number
from stdlib.is_instance import is_instance

# Gene types:
#       bool    supported as (int8, [0,1])
#       int,   numpy.int8/16/23/64, numpy.uint8/16/32/64
#       float, numpy.float16/32/64
#
GENE_INT_TYPES = (int, np.int8, np.int16, np.int32, np.int64)
GENE_FLOAT_TYPES = (float, np.float16, np.float32, np.float64)
GENE_TYPES = (bool,) + GENE_INT_TYPES + GENE_FLOAT_TYPES


class Solution:

    def __init__(self, gene_type: type, num_genes: int, gene_space=(0, 1)):
        assert is_instance(num_genes, int), "Invalid num_genes"
        assert gene_type in GENE_TYPES, "Invalid gene_type"

        # all genes must have the same type
        # all gene_spaces must be defined in the same way!
        #
        # bool       -> (int8, [0,1])
        # (min, max) -> dict(low=min, high=max)

        if gene_type == bool:
            gene_type = np.int8
            gene_space = [0, 1]

        if is_instance(gene_space, tuple[Number]):
                gene_space = dict(low=gene_space[0], high=gene_space[1])
        elif isinstance(gene_space, list) and not is_instance(gene_space, (list[Number])):
            assert len(gene_space) == num_genes, "Invalid gene_space length"
            for i in range(num_genes):
                gspace = gene_space[i]
                if is_instance(gspace, tuple[Number]):
                    gene_space[i] = dict(low=gspace[0], high=gspace[1])

        self._gene_type = gene_type
        self._num_genes = num_genes
        self._gene_space = gene_space
    # end

    @property
    def num_genes(self):
        return self._num_genes

    @property
    def gene_type(self):
        return self._gene_type

    @property
    def gene_space(self):
        return self._gene_space

    def population(self, sol_on_pop):
        return []

# end
