#
# https://github.com/YingxuH/genetic_algorithm/tree/master
#
import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm

x = np.linspace(0, 5, 1000)

ground_truth = x**3 - 2*(x**2) + 1


def func(a,b,c, d):
    return x**a - b*(x**2) + c

def fitness(params):
    return -np.sqrt(np.mean((ground_truth-func(**params))**2))

param_space = {
    "a": {"type": "float", "range":[0, 5]},
    "b": {"type": "float", "range":[-1, 5]},
    "c": {"type": "int", "range":[0, 3]},
    "d": {"type": "enum", "range": ["a","b","c"]}
}


def main():
    ga = GeneticAlgorithm(model=fitness,
                          param_space=param_space,
                          pop_size=100,
                          parent_pool_size=10,
                          keep_parent=False,
                          max_iter=100,
                          mutation_prob=0.3,
                          crossover_prob=0.7,
                          max_stop_rounds=5,
                          verbose=False)

    result = ga.evolve()
    print(result)

    predicted = func(**result["best params"])
    plt.scatter(x, ground_truth, s=3, label="ground truth")
    plt.scatter(x, predicted, s=3, c="r", label="predicted")
    plt.legend(loc="upper left")
    plt.show()



if __name__ == "__main__":
    main()