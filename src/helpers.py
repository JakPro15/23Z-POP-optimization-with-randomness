import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]

MAX_FUNCTION_CALLS = 150_000


def get_initial_population(size: int, dimensions: int, range_: float) -> list[Vector]:
    return [np.random.uniform(-1 * range_, range_, size=dimensions) for _ in range(size)]
