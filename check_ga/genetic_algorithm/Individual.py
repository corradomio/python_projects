import random


class Individual(object):
    def __init__(self, gene_set=None, fitness=None):
        self.gene_set = gene_set
        self.fitness = fitness

    @classmethod
    def _is_valid_operand(cls, other):
        return isinstance(other, Individual) and hasattr(other, "fitness")

    def set_gene_set(self, gene_set):
        self.gene_set = gene_set

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_gene_set(self):
        return self.gene_set

    def get_gene_set_to_print(self):
        to_print = {}
        for key, gene in self.gene_set.items():
            to_print[key] = gene.get_value()
        return to_print

    def get_fitness(self):
        return self.fitness

    def mutate(self, mutation_prob):
        # Out of mutation_prob, pick one parameter from the param dictionary and alter it by +-10% of its value
        # Mutation changes a single gene in each offspring randomly.
        for name, gene in self.gene_set.items():
            random_signal = random.random()
            if random_signal <= mutation_prob:
                gene.mutate()

    def __eq__(self, other):
        if not Individual._is_valid_operand(other):
            return NotImplemented
        else:
            return self.get_fitness() == other.get_fitness()

    def __lt__(self, other):
        if not Individual._is_valid_operand(other):
            return NotImplemented
        else:
            return self.get_fitness() < other.get_fitness()

    def __le__(self, other):
        if not Individual._is_valid_operand(other):
            return NotImplemented
        else:
            return self.get_fitness() <= other.get_fitness()

    def __gt__(self, other):
        if not Individual._is_valid_operand(other):
            return NotImplemented
        else:
            return self.get_fitness() > other.get_fitness()

    def __ge__(self, other):
        if not Individual._is_valid_operand(other):
            return NotImplemented
        else:
            return self.get_fitness() >= other.get_fitness()

    def __ne__(self, other):
        if not Individual._is_valid_operand(other):
            return NotImplemented
        else:
            return self.get_fitness() != other.get_fitness()




