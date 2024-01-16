import numpy as np
from numpy.typing import NDArray
from typing import Callable
from random import normalvariate

Vector = NDArray[np.float64]


def ackley(x: Vector) -> float:
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.square(x)))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + np.e + 20


def sphere(x: Vector) -> float:
    return np.sum(np.square(x))


def poly(x: Vector) -> float:
    return float(np.sum(np.power(x, 4) + 3 * np.power(x, 3) + 2 * np.square(x)))


def disturbed_square(x: Vector) -> float:
    return float(np.sum(np.square(x) / 100 + np.sin(x) + np.sin(2 * x)))


def himmelblau(x: Vector) -> float:
    return np.square(np.square(x[0]) + x[1] - 11) + np.square(x[0] + np.square(x[1]) - 7)


def check_bounds(x: float) -> float:
    return np.abs(x + 512 - 2048 * np.floor((x + 1024 + 512) / 2048)) - 512


def eggholder(x: Vector) -> float:
    x[0] = check_bounds(x[0])
    x[1] = check_bounds(x[1])
    return -1 * (x[1] + 47) * np.sin(np.sqrt(np.abs((x[0] / 2) + x[1] + 47))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - x[1] - 47)))


def add_randomness(fitness: Callable[[Vector], float], sigma1: float, sigma2: float) -> Callable[[Vector], float]:
    def fitness_with_randomness(x: Vector) -> float:
        zeta1 = np.ones_like(x) * normalvariate(0, sigma1)
        zeta2 = normalvariate(0, sigma2)
        return fitness(x + zeta1) + zeta2
    return fitness_with_randomness
