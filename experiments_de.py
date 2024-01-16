import numpy as np
from numpy.typing import NDArray
from functions import *
import random
import statistics
import sys
from differential_evolution import differential_evolution


Vector = NDArray[np.float64]


def get_initial_population(size: int, dimensions: int) -> list[Vector]:
    return [np.random.uniform(-1000, 1000, size=dimensions) for _ in range(size)]


def do_de_experiment(
    function: Callable[[Vector], float], sigma1: float, sigma2: float,
    dimensions: int, population_size: int, differential_weight: float, crossover_threshold: float
) -> tuple[float, float, float, float]:
    results: list[float] = []
    for seed_value in range(25):
        np.random.seed(seed_value)
        random.seed(seed_value)
        initial_population = get_initial_population(population_size, dimensions)
        randomized_function = add_randomness(function, sigma1, sigma2)
        result, _ = differential_evolution(initial_population, randomized_function, differential_weight,
                                    crossover_threshold, 500)
        result_fitness = function(result)
        results.append(result_fitness)
    average = statistics.mean(results)
    std_deviation = statistics.stdev(results)
    min_score = min(results)
    max_score = max(results)
    return average, std_deviation, min_score, max_score

if __name__ == "__main__":
    function_index = int(sys.argv[1])
    function = [ackley, sphere, poly, disturbed_square, himmelblau, eggholder][function_index]
    dimensions_options = [2] if function_index >= 4 else [10, 30, 50]
    file_name = f'experiment_results/differential_evolution_{function.__name__}.csv'

    counter = 0
    try:
        with open(file_name, 'r') as file:
            file_length = sum([1 for _ in file])
    except Exception:
        file_length = 0

    for sigma1 in [0, 1, 10]:
        for sigma2 in [0, 1, 10]:
            for dimensions in dimensions_options:
                for population_size in [10, 20, 30]:
                    for differential_weight in [0.4, 0.8, 1.4]:
                        for crossover_threshold in [0.0, 0.4, 0.8]:
                            counter += 1
                            if counter <= file_length:
                                continue

                            average, std_deviation, min_score, max_score =  do_de_experiment(
                                function, sigma1, sigma2, dimensions, population_size, differential_weight, crossover_threshold
                            )
                            with open(file_name, 'a+') as file:
                                file.write(
                                    f"{function.__name__},{sigma1},{sigma2},{dimensions},{population_size},{differential_weight}," + \
                                    f"{crossover_threshold},{average},{std_deviation},{min_score},{max_score}\n"
                                )