from genetic_algorithm.GeneInterface import GeneInterface
import random


class EnumGene(GeneInterface):
    def __init__(self, enum_values: list=None, gene_type=None):
        self.type = gene_type
        self.value = None
        self.enum_values = enum_values

    def init_value(self):
        self.mutate()

    def set_value(self, value):
        self.value = value

    def set_choices(self, enum_values):
        self.enum_values = enum_values

    def get_value(self):
        return self.value

    def get_type(self):
        return self.type

    def mutate(self):
        pos = random.randint(0, len(self.enum_values) - 1)
        new_value = self.enum_values[pos]

        if new_value == self.value:
            self.mutate()
        else:
            self.value = new_value

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result