import random
from genetic_algorithm.GeneInterface import GeneInterface


class IntGene(GeneInterface):
    def __init__(self, value=None, lower_bound=None, upper_bound=None, gene_type=None):
        self.type = gene_type
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def init_value(self):
        self.mutate()

    def set_value(self, value):
        self.value = value

    def set_boundary(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_value(self):
        return self.value

    def get_type(self):
        return self.type

    def mutate(self):
        new_value = random.randint(self.lower_bound, self.upper_bound)
        if new_value == self.value:
            self.mutate()
        else:
            self.value = new_value

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


