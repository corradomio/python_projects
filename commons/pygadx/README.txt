Fitness Function
----------------

    fitness_func(ga_instance, solution, solution_idx [, object_instance])
        ga_instance: current GA object executing the function
        solution: candidate solution
                multiple solutions if 'fitness_batch_size > 1'
        solution_idx: index assigned to the solution, member of a population
                solution indices   if 'fitness_batch_size > 1'
        object_instance: if the function is an object's method


the function can be an object method

if the fitness function returns a numeric value, then the problem is single-objective.
if the fitness function returns a list, tuple, or numpy.ndarray, then the problem is multi-objective


Initial Population
------------------

    Two possibilites:

    1) constructor's parameter 'initial_population'
    2) 'sol_per_pop', 'num_genes',
       'init_range_low', 'init_range_high':


Gene types
----------

    1) single type: all genes have the same type
    2) a type for each gene


GA constructor
--------------

    num_generations: Number of generations.
    num_parents_mating: Number of solutions to be selected as parents.

    fitness_func: Accepts a function/method and returns the fitness value(s) of the solution
    fitness_batch_size=None: A new optional parameter called fitness_batch_size is supported to calculate the fitness function in batches.

    initial_population: A user-defined initial population.

    sol_per_pop: Number of solutions (i.e. chromosomes) within the population.

    allow_duplicate_genes=True,
    delay_after_gen=0.0,

    -----------------
    num_genes: Number of genes in the solution/chromosome.

    gene_type=float: Controls the gene type.
        It can be assigned to a single data type that is applied to all genes or can specify the data type of each
            individual gene.
        It defaults to float which means all genes are of float data type.
        The gene_type parameter can be assigned to a numeric value of any of these types: int, float, and
            numpy.int/uint/float(8-64).
        It can be assigned to a list, tuple, or a numpy.ndarray which hold a data type for each gene
            (e.g. gene_type=[int, float, numpy.int8]). This helps to control the data type of each individual gene.
        A precision for the float data types can be specified (e.g. gene_type=[float, 2].

    gene_space=None: It is used to specify the possible values for each gene in case the user wants to restrict the gene values.
        It is useful if the gene space is restricted to a certain range or to discrete values.
        Discrete space (enum): list, range, numpy.ndarray.
        Continuous space: dict(min=..,max=..)
        If each gene has its own space, then the gene_space parameter can be nested like [[0.4, -5], [0.5, -3.2, 8.2, -9], ...]
            where the first sublist determines the values for the first gene, the second sublist for the second gene,
            and so on.
        [NO] If the nested list/tuple has a None value, then the gene’s initial value is selected randomly from the range
            specified by the 2 parameters init_range_low and init_range_high and its mutation value is selected randomly
            from the range specified by the 2 parameters random_mutation_min_val and random_mutation_max_val.

    [NO] init_range_low=-4: The lower value of the random range from which the gene values in the initial population are selected.
    [NO] init_range_high=4: The upper value of the random range from which the gene values in the initial population are selected.

    -----------------

    parent_selection_type="sss",
    keep_parents=-1,
    keep_elitism=1,

    K_tournament=3,

    crossover_type="single_point",
    crossover_probability=None,

    mutation_type="random",
    mutation_probability=None,
    mutation_by_replacement=False,
    mutation_percent_genes='default',
    mutation_num_genes=None,

    random_mutation_min_val=-1.0,
    random_mutation_max_val=1.0,

    on_start=None,
    on_fitness=None,
    on_parents=None,
    on_crossover=None,
    on_mutation=None,
    on_generation=None,
    on_stop=None,

    save_best_solutions=False,
    save_solutions=False,

    stop_criteria=None,
    parallel_processing=None,

    random_seed=None,
    suppress_warnings=False,
    logger=None


PyGAD Modules
-------------

    pygad
    pygad.nn
    pygad.cnn
    pygad.gann
    pygad.gacnn
    pygad.torchga
    pygad.visualize
    pygad.utils
    pygad.helper


Supported types
---------------

    int,   numpy.int8/16/23/64, numpy.uint8/16/32/64
    float, numpy.float16/32/64

Range types
-----------

    None
    range -> list(range(...))
