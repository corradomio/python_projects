# import numpy as np
import pandas as pd

from genetic_algorithm.FloatGene import FloatGene
from genetic_algorithm.IntGene import IntGene
from genetic_algorithm.ObjectGene import ObjectGene
from genetic_algorithm.Individual import Individual
from genetic_algorithm.Generation import Generation


class GeneticAlgorithm(object):
    def __init__(self, model, param_space, pop_size=100, parent_pool_size=10, keep_parent=False,
                 selection_method="rank", max_iter=100, crossover_prob=0.7, mutation_prob=0.3,
                 max_stop_rounds=5, verbose=True):
        self.model = model
        self.param_space = param_space
        self.pop_size = pop_size
        self.parent_pool_size = parent_pool_size
        self.keep_parent = keep_parent
        self.selection_method = selection_method
        self.max_iter = max_iter
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_stop_rounds = max_stop_rounds
        self.verbose = verbose

        self.still_count = 0
        self.iteration = 0
        self.last_best_fitness = None
        self.best_fitness = None
        self.best_parameters = None

    def populate(self):
        inds = []
        for i in range(self.pop_size):
            gene_set = {}
            for key, value in self.param_space.items():
                current_gene = None
                if value["type"] == "int":
                    current_gene = IntGene()
                    current_gene.set_boundary(value['range'][0], value['range'][1])
                    current_gene.init_value()
                elif value["type"] == "float":
                    current_gene = FloatGene()
                    current_gene.set_boundary(value['range'][0], value['range'][1])
                    current_gene.init_value()
                elif value["type"] == "enum":
                    current_gene = ObjectGene()
                    current_gene.set_choices(value['range'])
                    current_gene.init_value()
                elif value["type"] == "object":
                    current_gene = ObjectGene()
                    current_gene.set_choices(value['range'])
                    current_gene.init_value()
                gene_set[key] = current_gene
            current_individual = Individual(gene_set=gene_set)
            inds.append(current_individual)
        return Generation(model=self.model, inds=inds, parent_pool_size=self.parent_pool_size,
                          pop_size=self.pop_size, arg_lst=self.param_space.keys())

    def evolve(self):
        best_table = []
        current_gen = self.populate()

        while self.still_count < self.max_stop_rounds and self.iteration < self.max_iter:

            current_gen.select(self.selection_method)

            best_table.append([current_gen.get_best_fitness(),
                               current_gen.best_gene_set_to_print()])

            if self.verbose:
                # The best result in the current iteration.
                print("Best fitness : {} with params: {}".format(current_gen.get_best_fitness(),
                                                                 current_gen.best_gene_set_to_print()))

            if self.best_fitness is None or current_gen.get_best_fitness() >= self.best_fitness:
                self.best_fitness = current_gen.get_best_fitness()
                self.best_parameters = current_gen.best_gene_set_to_print()

            if current_gen.get_best_fitness() == self.last_best_fitness:
                self.still_count += 1
            else:
                self.still_count = 0

            self.iteration += 1
            self.last_best_fitness = current_gen.get_best_fitness()

            current_gen = current_gen.crossover(self.crossover_prob, self.keep_parent)
            current_gen.mutate(self.mutation_prob)

        best_table = pd.DataFrame(best_table, columns=["Fitness", "Params"])

        return {"best fitness": self.last_best_fitness,
                "best params": self.best_parameters,
                "history": best_table}



















