import random
from genetic_algorithm.GeneInterface import GeneInterface


class ObjectGene(GeneInterface):
    def __init__(self, choices=None, gene_type=None):
        self.type = gene_type
        self.value = None
        self.choices = choices

    def init_value(self):
        self.mutate()

    def set_value(self, value):
        self.value = value

    def set_choices(self, choices):
        self.choices = choices

    def get_value(self):
        return self.value

    def get_type(self):
        return self.type

    def mutate(self):
        new_value = random.choice(self.choices)
        if new_value == self.value:
            self.mutate()
        else:
            self.value = new_value

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result



