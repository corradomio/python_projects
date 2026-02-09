import random
from copy import copy
from scipy.special import softmax
from genetic_algorithm.Individual import Individual


class Generation(object):
    def __init__(self, model=None, inds=None, parent_pool_size=None, pop_size=None, arg_lst=None):
        self.model = model
        # expecting list
        self.inds = inds
        self.parent_pool_size = parent_pool_size
        self.pop_size = pop_size
        self.arg_lst = list(arg_lst)

        self.parents = None
        self.best_ind = None

    def set_model(self, model):
        self.model = model

    def set_inds(self, inds):
        self.inds = inds

    def get_model(self):
        return self.model

    def get_inds(self):
        return self.inds

    def get_best_ind(self):
        return self.best_ind

    def get_best_fitness(self):
        return self.best_ind.get_fitness()

    def get_best_gene_set(self):
        return self.best_ind.get_gene_set()

    def best_gene_set_to_print(self):
        return self.best_ind.get_gene_set_to_print()

    def calculate_fitness(self):
        for ind in self.inds:
            fitness = self.model(ind.get_gene_set_to_print())
            ind.set_fitness(fitness)

    def select(self, selection_method):
        self.calculate_fitness()
        if selection_method == "rank":
            self.inds.sort(reverse=True)
            self.parents = self.inds[:self.parent_pool_size]
            self.best_ind = self.inds[0]
        elif selection_method == "roulette_wheel":
            fitnesses = softmax([ind.get_fitness() for ind in self.inds])
            self.parents = random.choices(self.inds, weights=fitnesses, k=self.parent_pool_size)
            self.best_ind = sorted(self.parents, reverse=True)[0]
        else:
            raise Exception("unknown selection method: {}".format(selection_method))

    # TODO: shift implementation to individual
    # uniform cross over
    def crossover(self, crossover_prob, keep_parent=True):
        if self.parents is None:
            print("generation haven't been selected")
            self.select()

        offspring = []
        if keep_parent:
            offspring.extend(self.parents)

        for k in range((self.pop_size-len(offspring))//2 + 1):
            # Parents to mate.
            parent1 = self.parents[k % self.parent_pool_size]
            parent2 = self.parents[(k + 1) % self.parent_pool_size]

            random_signal = random.random()
            if random_signal <= crossover_prob:
                # randomly select half of the parameters and set to zeros.
                args_parent1_gene = random.sample(self.arg_lst, len(self.arg_lst) // 2)

                # TODO: shift to individual implementation
                parent1_gene1 = {key: copy(value) for key, value in parent1.get_gene_set().items() if key in args_parent1_gene}
                parent1_gene2 = {key: copy(value) for key, value in parent1.get_gene_set().items() if key not in args_parent1_gene}

                # select the other half of the parameters and set to zeros.
                parent2_gene1 = {key: copy(value) for key, value in parent2.get_gene_set().items() if key not in args_parent1_gene}
                parent2_gene2 = {key: copy(value) for key, value in parent2.get_gene_set().items() if key in args_parent1_gene}

                # create current offsprint as the summation of the two parents.
                child_gene1 = {**parent1_gene1, **parent2_gene1}
                child_gene2 = {**parent1_gene2, **parent2_gene2}

                child_1 = Individual(gene_set=child_gene1)
                child_2 = Individual(gene_set=child_gene2)
            else:
                child_1, child_2 = parent1, parent2

            offspring.append(child_1)
            offspring.append(child_2)

        offspring = offspring[:self.pop_size]

        return Generation(model=self.model, inds=offspring, parent_pool_size=self.parent_pool_size,
                          pop_size=self.pop_size, arg_lst=self.arg_lst)

    def mutate(self, mutation_prob):
        for ind in self.inds:
            ind.mutate(mutation_prob)



