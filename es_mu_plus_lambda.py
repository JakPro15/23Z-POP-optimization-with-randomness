import numpy as np
from numpy.typing import NDArray
from typing import Callable
from random import choices, normalvariate, shuffle, uniform
from copy import deepcopy

Vector = NDArray[np.float64]


def do_mutation(population: list[tuple[Vector, Vector]]) -> list[tuple[Vector, Vector]]:
    for element, mutation_strength in population:
        a = normalvariate(0, 1)
        b = np.random.normal(0, 1, size=element.shape)
        mutation_strength *= np.exp(a / np.sqrt(2 * len(element)) + b / np.sqrt(2 * np.sqrt(len(element))))
        assert isinstance(mutation_strength, np.ndarray)
        element += mutation_strength * np.random.normal(0, 1, size=element.shape)
    return population


def do_crossover(population: list[tuple[Vector, Vector]]) -> list[tuple[Vector, Vector]]:
    parents = deepcopy(population)
    shuffle(parents)
    result: list[tuple[Vector, Vector]] = []
    for i in range(len(parents)):
        weight = uniform(0, 1)
        result.append((weight * parents[i // 2][0] + weight * parents[i // 2 + 1][0],
                       weight * parents[i // 2][1] + weight * parents[i // 2 + 1][1]))
    return result


def do_succession(
    old_population: list[tuple[Vector, Vector, float]], new_population: list[tuple[Vector, Vector, float]]
) -> list[tuple[Vector, Vector, float]]:
    summed_populations = old_population + new_population
    returned_population: list[tuple[Vector, Vector, float]] = []
    for _ in old_population:
        index, best_element = min(list(enumerate(summed_populations)), key=lambda x: x[1][2])
        returned_population.append(best_element)
        summed_populations.pop(index)
    return returned_population


def es_mu_plus_lambda(initial_population: list[Vector], fitness: Callable[[Vector], float], lambd: int,
                      intial_mutation_strength: float, max_iterations: int) -> Vector:
    iteration = 0
    population = [(element, np.ones_like(element, dtype=np.float64) * intial_mutation_strength, fitness(element))
                  for element in initial_population]
    best_element, _, best_fitness = min(population, key=lambda x: x[2])

    while iteration < max_iterations:
        reproduction_population = [element[:2] for element in choices(population, k=lambd)]
        crossed_over_population = do_crossover(reproduction_population)
        mutated_population = do_mutation(crossed_over_population)

        new_population = [(element[0], element[1], fitness(element[0])) for element in mutated_population]
        new_best_element, _, new_best_fitness = min(new_population, key=lambda x: x[2])
        if new_best_fitness < best_fitness:
            best_element, best_fitness = new_best_element, new_best_fitness

        population = [(element, mutation_strength, fitness(element)) for element, mutation_strength, _ in population]
        population = do_succession(population, new_population)
        iteration += 1

    return best_element
