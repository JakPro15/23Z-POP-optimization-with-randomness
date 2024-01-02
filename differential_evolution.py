import numpy as np
from numpy.typing import NDArray
from typing import Callable
from random import choices, uniform

Vector = NDArray[np.float64]


def do_crossover(element_1: Vector, element_2: Vector, threshold: float) -> Vector:
    assert element_1.size == element_2.size
    result = np.copy(element_1)
    for dimension in range(element_1.size):
        if uniform(0, 1) < threshold:
            result[dimension] = element_2[dimension]
    return result


def diff_evo(initial_population: list[Vector], fitness: Callable[[Vector], float],
             differential_weight: float, crossover_threshold: float, max_iterations: int) -> Vector:
    iteration = 0
    population = initial_population
    while iteration < max_iterations:
        next_population = []
        best_element = min(population, key=lambda el: fitness(el))
        for element in population:
            rand_element_1, rand_element_2 = choices(population, k=2)
            mutation_element = best_element + differential_weight * \
                (rand_element_1 - rand_element_2)
            crossover_element = do_crossover(
                element, mutation_element, crossover_threshold)
            next_population.append(
                min([element, crossover_element], key=lambda el: fitness(el)))
        population = next_population
        iteration += 1

    return max(population, key=lambda el: fitness(el))
