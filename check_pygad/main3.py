import pygadx as pygad
import numpy

from pygadx import Solution


class Problem(pygad.Solution):

    def __init__(self):
        super().__init__(float, 6, (-2, 5))
        self.desired_output = 44
        self.function_inputs = numpy.array([4, -2, 3.5, 5, -11, -4.7])

    # @property
    # def num_genes(self):
    #     return len(self.function_inputs)

    # @property
    # def gene_type(self):
    #     return [float]*self.num_genes

    # @property
    # def gene_space(self):
    #     return dict(low=-2, high=5)

    def fitness_func(self, ga_instance, solution, solution_idx):
        output = numpy.sum(solution * self.function_inputs)
        fitness = 1.0 / numpy.abs(output - self.desired_output)
        return fitness


ff = pygad.FitnessFunction

problem = Problem()
solution = Solution(float, 6, (-2, 5))

# fitness_func = problem.fitness_func

num_generations = 20
num_parents_mating = 4

sol_per_pop = 8
# num_genes = problem.num_genes
# gene_type = problem.gene_type

# init_range_low = -2
# init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(
    solution=solution,

    # num_genes=problem.num_genes,
    # gene_type=problem.gene_type,
    # gene_space=problem.gene_space,

    # init_range_low=init_range_low,
    # init_range_high=init_range_high,

    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=problem.fitness_func,
    sol_per_pop=sol_per_pop,

    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
    suppress_warnings=True)

ga_instance.run()
ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = numpy.sum(problem.function_inputs * solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
