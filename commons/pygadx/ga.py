from typing import Optional, Literal

import pygad

from main2 import num_genes


# ---------------------------------------------------------------------------
# Callbacks: from function to method
# ---------------------------------------------------------------------------

def _on_start(self):
    self._on_start()
    pass


def _on_fitness(self, population_fitness):
    self._on_fitness(population_fitness)
    pass


def _on_parents(self, selected_parents):
    self._on_parents(selected_parents)
    pass


def _on_crossover(self, offspring_crossover):
    self._on_crossover(offspring_crossover)
    pass


def _on_mutation(self, offspring_mutation):
    self._on_mutation(offspring_mutation)
    pass


def _on_generation(self):
    self._on_generation()
    pass


def _on_stop(self, last_population_fitness):
    self._on_stop(last_population_fitness)
    pass


class GA(pygad.GA):
    def __init__(self, solution=None, **kwargs):
        if solution is not None:
            kwargs = kwargs | dict(
                gene_type=solution.gene_type,
                num_genes=solution.num_genes,
                gene_space=solution.gene_space,
            )
        super().__init__(
            on_start=_on_start,
            on_fitness=_on_fitness,
            on_parents=_on_parents,
            on_crossover=_on_crossover,
            on_mutation=_on_mutation,
            on_generation=_on_generation,
            on_stop=_on_stop,
            **kwargs)
        pass

    # -----------------------------------------------------------------------
    # Overrides
    # -----------------------------------------------------------------------

    def initialize_population(self, low, high, allow_duplicate_genes, mutation_by_replacement, gene_type):
        return super().initialize_population(low, high, allow_duplicate_genes, mutation_by_replacement, gene_type)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    # Note: pygad.GA contains the members 'self.on_start', ...

    def _on_start(self):
        """a function/method to be called only once before the genetic algorithm starts its evolution"""
        pass

    def _on_fitness(self, population_fitness):
        """a function/method to be called after calculating the fitness values of all solutions in the population"""
        pass

    def _on_parents(self, selected_parents):
        """a function/method to be called after selecting the parents that mates"""
        pass

    def _on_crossover(self, offspring_crossover):
        """a function to be called each time the crossover operation is applied"""
        pass

    def _on_mutation(self, offspring_mutation):
        """a function to be called each time the mutation operation is applied"""
        pass

    def _on_generation(self) -> Optional[Literal["stop"]]:
        """a function to be called after each generation.
         If the function returned the string 'stop', then the run() method stops without completing the other
         generations"""
        pass

    def _on_stop(self, last_population_fitness):
        """a function to be called only once exactly before the genetic algorithm stops or when it completes all the
        generations"""
        pass

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end
