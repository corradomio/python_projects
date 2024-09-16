# from random import choice, random, randrange
# from typing import cast
#
# import numpy as np
#
# from stdlib.is_instance import is_instance
#
# # Gene types:
# #       bool
# #       int,   numpy.int8/16/23/64, numpy.uint8/16/32/64
# #       float, numpy.float16/32/64
# #
# GENE_INT_TYPES = (int, np.int8, np.int16, np.int32, np.int64)
# GENE_FLOAT_TYPES = (float, np.float16, np.float32, np.float64)
# GENE_TYPES = (bool,) + GENE_INT_TYPES + GENE_FLOAT_TYPES
#
# # gene_range:
# #   (min, max)      range min included, max excluded
# #   [v1, v2,...]    enumeration
# #   (gene_range_1, ...) | [gene_range_1, ...]
#
# class Range[T]:
#
#     def random(self) -> T:
#         pass
#
#     def is_valid(self, x: T) -> bool:
#         pass
#
# class MinMaxRange(Range):
#     def __init__(self, minmax):
#         if isinstance(minmax, dict):
#             min = minmax['low']
#             max = minmax['high']
#         else:
#             min, max = minmax
#         self._min = min
#         self._max = max
#         self._delta = max - min
#         if type(min) in [float]:
#             self._rand = lambda : self._min + random()*self._delta
#         else:
#             self._rand = lambda : randrange(self._min, self._max)
#
#     @property
#     def low(self):
#         return self._min
#
#     @property
#     def high(self):
#         return self._max
#
#     def random(self):
#         return self._rand()
#
#     def is_valid(self, x) -> bool:
#         return self._min <= x < self._max
#
#
# class EnumRange(Range):
#     def __init__(self, enum):
#         self._enum = enum
#
#     def random(self):
#         return choice(self._enum)
#
#     def is_valid(self, v):
#         return v in self._enum
#
#
# class Solution:
#
#     def __init__(self, gene_type: type, num_genes: int, gene_space=(0, 1)):
#         assert is_instance(num_genes, int), "Invalid num_genes"
#         assert gene_type in GENE_TYPES, "Invalid gene_type"
#
#         # all genes must have the same type
#         # all gene_spaces must defined in the same way!
#         # (min, max) -> dict(low=min, high=max)
#
#         if gene_type == bool:
#             gene_type = np.int8
#             gene_space = [0, 1]
#
#         self._gene_type = gene_type
#         self._num_genes = num_genes
#         self._gene_space = gene_space
#         self._validate : list[Range] = []
#
#         is_float = gene_type in GENE_FLOAT_TYPES
#         is_single_space = is_instance(gene_space, (tuple[int], tuple[float], list[int], list[float], dict))
#         assert is_single_space or len(gene_space) == num_genes, "Invalid gene_space"
#
#         if is_single_space:
#             if is_instance(gene_space, tuple):
#                 gene_space = dict(low=gene_space[0], high=gene_space[1])
#         else:
#             gene_space = gene_space[:]
#             for i in range(len(gene_space)):
#                 if is_instance(gene_space[i], tuple):
#                     gene_space[i] = dict(low=gene_space[0], high=gene_space[1])
#
#         gspace = gene_space if is_single_space else gene_space[0]
#         is_mimax = isinstance(gspace, (tuple,dict))
#
#         if is_single_space:
#             if not is_mimax:
#                 validate = [EnumRange(gene_space) for i in range(num_genes)]
#             else:
#                 validate = [MinMaxRange(gene_space) for i in range(num_genes)]
#         else:
#             validate: list[Range] = cast(list[Range], [None]*num_genes)
#             if not is_mimax:
#                 for i in range(num_genes):
#                     validate[i] = EnumRange(gene_space[i])
#             else:
#                 for i in range(num_genes):
#                     validate[i] = MinMaxRange(gene_space[i])
#
#         self._is_single_space = is_single_space
#         self._type_mode = "float" if is_float else "int" if is_mimax else "enum"
#         self._validate = validate
#     # end
#
#     @property
#     def num_genes(self):
#         return self._num_genes
#
#     @property
#     def gene_type(self):
#         return self._gene_type
#
#     @property
#     def gene_space(self):
#         return self._gene_space
#
#     def is_valid(self, sol):
#         num_genes = self.num_genes
#         assert len(sol) == num_genes, "Invalid solution"
#         for i in range(num_genes):
#             if not self._validate[i].is_valid(sol[i]):
#                 return False
#         return True
#
#     def population(self, sol_per_pop) -> np.ndarray:
#         num_genes = self._num_genes
#         if self._is_single_space and self._type_mode == "float":
#             mmr: MinMaxRange = cast(MinMaxRange, self._validate[0])
#             pop = np.random.uniform(low=mmr.low, high=mmr.high, size=(sol_per_pop, num_genes))
#         else:
#             pop = [[self._validate[i].random() for i in range(num_genes)]
#                    for s in range(sol_per_pop)]
#             pop = np.array(pop, dtype=self.gene_type)
#         return pop
#     # end
