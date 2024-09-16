from typing import Any

from pygad import GA


class FitnessFunction:

    def __call__(self, ga_instance: GA, solution: Any, solution_idx: int):
        return self.fitness_func(ga_instance, solution, solution_idx)

    def fitness_func(self, ga_instance: GA, solution: Any, solution_idx: int):
        return self.fitness(solution)

    def fitness(self, solution):
        pass

# end
