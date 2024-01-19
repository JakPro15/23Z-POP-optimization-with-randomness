import numpy as np
from numpy.typing import NDArray
from typing import Callable
from random import choices, normalvariate, shuffle, uniform

Vector = NDArray[np.float64]


def do_mutation(population: list[tuple[Vector, Vector]]) -> list[tuple[Vector, Vector]]:
    for element, mutation_strength in population:
        a = normalvariate(0, 1)
        b = np.random.normal(0, 1, size=element.shape)
        mutation_strength *= np.exp(a / np.sqrt(2 * np.sqrt(len(element))) + b / np.sqrt(2 * len(element)))
        element += mutation_strength * np.random.normal(0, 1, size=element.shape)
    return population


def do_crossover(population: list[tuple[Vector, Vector]]) -> list[tuple[Vector, Vector]]:
    parents = population.copy()
    shuffle(parents)
    result: list[tuple[Vector, Vector]] = []
    for i in range(len(parents)):
        weight = uniform(0, 1)
        result.append((weight * parents[2 * (i // 2)][0] + (1 - weight) * parents[2 * (i // 2) + 1][0],
                       weight * parents[2 * (i // 2)][1] + (1 - weight) * parents[2 * (i // 2) + 1][1]))
    return result


def do_succession(
    old_population: list[tuple[Vector, Vector, float]], new_population: list[tuple[Vector, Vector, float]]
) -> list[tuple[Vector, Vector, float]]:
    summed_populations = old_population + new_population
    sorted_populations = sorted(summed_populations, key=lambda x: x[2])
    return sorted_populations[:len(old_population)]


def es_mu_plus_lambda(
    initial_population: list[Vector], fitness: Callable[[Vector], float], lambd: int,
    intial_mutation_strength: float, max_iterations: int
) -> tuple[Vector, list[Vector]]:
    iteration = 0
    population = [(element, np.ones_like(element, dtype=np.float64) * intial_mutation_strength, fitness(element))
                  for element in initial_population]
    best_element, _, _ = min(population, key=lambda x: x[2])
    best_element = best_element.copy()

    best_elements: list[Vector] = []

    while iteration < max_iterations:
        reproduction_population = [element[:2] for element in choices(population, k=lambd)]
        crossed_over_population = do_crossover(reproduction_population)
        mutated_population = do_mutation(crossed_over_population)

        new_population = [(element[0], element[1], fitness(element[0])) for element in mutated_population]
        population = [(element, mutation_strength, fitness(element)) for element, mutation_strength, _ in population]

        population = do_succession(population, new_population)
        best_elements.append(min(population, key=lambda x: x[2])[0])
        iteration += 1

    return best_element, best_elements

# mu + (mu + lambda) * maxiter = calls
# maxiter = (calls - mu) // (mu + lambda)
