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


def differential_evolution(
    initial_population: list[Vector], fitness: Callable[[Vector], float],
    differential_weight: float, crossover_threshold: float, max_iterations: int
) -> tuple[Vector, list[Vector]]:
    iteration = 0
    best_elements: list[Vector] = []

    population_fitnesses = [(element, fitness(element))
                            for element in initial_population]
    while iteration < max_iterations:
        next_population_fitnesses: list[tuple[Vector, float]] = []

        best_element = min(population_fitnesses, key=lambda el: el[1])
        best_elements.append(best_element[0])
        for element in population_fitnesses:
            rand_element_1, rand_element_2 = choices(population_fitnesses, k=2)
            mutation_element = best_element[0] + differential_weight * \
                (rand_element_1[0] - rand_element_2[0])
            crossover_element = do_crossover(
                element[0], mutation_element, crossover_threshold)
            crossover_element = (crossover_element, fitness(crossover_element))
            next_population_fitnesses.append(
                min(element, crossover_element, key=lambda el: el[1]))
        population_fitnesses = [(el, fitness(el))
                                for el, _ in next_population_fitnesses]
        iteration += 1

    best_element = min(population_fitnesses, key=lambda el: el[1])[0]
    best_elements.append(best_element)

    return best_element, best_elements
