import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


def get_initial_population(size: int, dimensions: int) -> list[Vector]:
    return [np.random.uniform(-1000, 1000, size=dimensions) for _ in range(size)]
