import numpy as np
from numpy.typing import NDArray

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


def eggholder(x: Vector) -> float:
    return -1 * (x[1] + 47) * np.sin(np.sqrt(np.abs((x[0] / 2) + x[1] + 47))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - x[1] - 47)))
