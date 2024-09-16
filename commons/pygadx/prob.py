class Problem:

    def __init__(self):
        pass

    def fitness_func(self, ga_instance, solution, solution_idx):
        return self.fitness(solution)

    def fitness(self, solution) -> float:
        pass

