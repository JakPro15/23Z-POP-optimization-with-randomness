import numpy as np
from numpy.typing import NDArray
from typing import Callable
from random import choices

Vector = NDArray[np.float64]


def do_mutation(population: list[tuple[Vector, float]]) -> list[tuple[Vector, float]]:
    ...


def do_crossover(population: list[tuple[Vector, float]]) -> list[tuple[Vector, float]]:
    ...


def do_succession(
    old_population: list[tuple[Vector, float, float]], new_population: list[tuple[Vector, float, float]]
) -> list[tuple[Vector, float, float]]:
    ...


def es_mu_plus_lambda(initial_population: list[Vector], fitness: Callable[[Vector], float], lambd: int,
                      intial_mutation_strength: float, max_iterations: int) -> Vector:
    iteration = 0
    population = [(element, intial_mutation_strength, fitness(element)) for element in initial_population]
    best_element, _, best_fitness = max(population, key=lambda x: x[1])

    while iteration < max_iterations:
        reproduction_population = [element[:2] for element in choices(population, k=lambd)]
        mutated_population = do_mutation(reproduction_population)
        crossed_over_population = do_crossover(mutated_population)

        new_population = [(element[0], element[1], fitness(element[0])) for element in crossed_over_population]
        new_best_element, _, new_best_fitness = max(new_population, key=lambda x: x[1])
        if new_best_fitness > best_fitness:
            best_element, best_fitness = new_best_element, new_best_fitness

        population = do_succession(population, new_population)
        iteration += 1

    return best_element
