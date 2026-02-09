from genetic_algorithm.GeneInterface import GeneInterface
import random


class FloatGene(GeneInterface):
    def __init__(self, lower_bound=None, upper_bound=None, gene_type=None):
        self.type = gene_type
        self.value = None
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
        if self.value is None:
            new_value = random.uniform(self.lower_bound, self.upper_bound)
        else:
            new_value = self.value * (1 + random.uniform(-0.2, 0.2))
            new_value = max(self.lower_bound, min(self.upper_bound, new_value))

        if new_value == self.value:
            self.mutate()
        else:
            self.value = new_value

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result



