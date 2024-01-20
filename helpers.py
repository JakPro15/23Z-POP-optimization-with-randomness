import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]

MAX_FUNCTION_CALLS = 150_000


def get_initial_population(size: int, dimensions: int) -> list[Vector]:
    return [np.random.uniform(-512, 512, size=dimensions) for _ in range(size)]
