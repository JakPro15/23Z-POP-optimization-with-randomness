import numpy as np
from numpy.typing import NDArray
from functions import *
import random
import statistics
import sys
from es_mu_plus_lambda import es_mu_plus_lambda
from helpers import get_initial_population, MAX_FUNCTION_CALLS, Vector


def do_es_experiment(
    function: Callable[[Vector], float], sigma1: float, sigma2: float,
    dimensions: int, mu: int, lambd: int, initial_mutation_strength: float,
    max_iterations: int
) -> tuple[float, float, float, float]:
    results: list[float] = []
    for seed_value in range(25):
        np.random.seed(seed_value)
        random.seed(seed_value)
        if function == ackley:
            range_ = 32
        else:
            range_ = 512
        initial_population = get_initial_population(mu, dimensions, range_)
        randomized_function = add_randomness(function, sigma1, sigma2)
        result, _ = es_mu_plus_lambda(initial_population, randomized_function, lambd,
                                      2 * range_ * initial_mutation_strength, max_iterations)
        result_fitness = function(result)
        results.append(result_fitness)
    average = statistics.mean(results)
    std_deviation = statistics.stdev(results)
    min_score = min(results)
    max_score = max(results)
    return average, std_deviation, min_score, max_score


if __name__ == "__main__":
    function_index = int(sys.argv[1])
    function = [ackley, sphere, poly, disturbed_square,
                himmelblau, eggholder][function_index]
    dimensions_options = [2] if function_index >= 4 else [10, 30, 50]
    file_name = f'experiment_results/es_mu_plus_lambda_{function.__name__}.csv'

    counter = 0
    try:
        with open(file_name, 'r') as file:
            file_length = sum([1 for _ in file])
    except Exception:
        file_length = 0

    for sigma1 in [1, 10]:
        for sigma2 in [1, 10]:
            for dimensions in dimensions_options:
                for mu in [10, 20, 30]:
                    for lambd in [5 * mu, 7 * mu, 9 * mu]:
                        for initial_mutation_strength in [0.02, 0.1, 0.5]:
                            counter += 1
                            if counter <= file_length:
                                continue

                            max_iterations = (
                                MAX_FUNCTION_CALLS - mu) // (mu + lambd)
                            average, std_deviation, min_score, max_score = do_es_experiment(
                                function, sigma1, sigma2, dimensions, mu, lambd, initial_mutation_strength, max_iterations
                            )
                            with open(file_name, 'a+') as file:
                                file.write(
                                    f"{function.__name__},{sigma1},{sigma2},{dimensions},{mu},{lambd},{initial_mutation_strength},{max_iterations}," +
                                    f"{average},{std_deviation},{min_score},{max_score}\n"
                                )
