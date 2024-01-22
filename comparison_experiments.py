import numpy as np
import random
from functions import *
from helpers import get_initial_population, Vector
from differential_evolution import differential_evolution
from es_mu_plus_lambda import es_mu_plus_lambda
from typing import Callable
import pickle
from multiprocessing import Manager, Pool


def get_plot_data(best_de: list[tuple[int, float, float, int]],
                  best_es_mu: list[tuple[int, int, float, int]],
                  params: list[tuple[Callable[[Vector], float], float, float, int]],
                  curves_de: list[None | np.ndarray],
                  curves_es_mu: list[None | np.ndarray],
                  calls_de: list[None | np.ndarray],
                  calls_es_mu: list[None | np.ndarray],
                  best_points_de: list[list[Vector]],
                  best_points_es_mu: list[list[Vector]],
                  param_start: int,
                  param_end: int
                  ):

    SEED_COUNT = 25

    for i in range(param_start, min(param_end, len(params))):

        function, sigma1, sigma2, dimensions = params[i]
        pop_size, diff_weight, cross_thresh, max_iter_de = best_de[i]
        mu, lambd, init_mut_strength, max_iter_es_mu = best_es_mu[i]

        results_de = []
        results_es_mu = []
        for seed_value in range(SEED_COUNT):
            np.random.seed(seed_value)
            random.seed(seed_value)
            randomized_function = add_randomness(
                function, sigma1, sigma2)

            if function == ackley:
                initial_population_de = get_initial_population(
                    pop_size, dimensions, 32)
            else:
                initial_population_de = get_initial_population(
                    pop_size, dimensions, 512)

            point_de, result_de = differential_evolution(
                initial_population_de, randomized_function, diff_weight, cross_thresh, max_iter_de)

            results_de.append(np.array([function(el) for el in result_de]))
            best_points_de[i].append(point_de)

            np.random.seed(seed_value)
            random.seed(seed_value)

            if function == ackley:
                initial_population_es_mu = get_initial_population(
                    mu, dimensions, 32)
                init_mut_strength *= 64
            else:
                initial_population_es_mu = get_initial_population(
                    mu, dimensions, 512)
                init_mut_strength *= 1024

            point_es_mu, result_es_mu = es_mu_plus_lambda(
                initial_population_es_mu, randomized_function, lambd, init_mut_strength, max_iter_es_mu)

            results_es_mu.append(
                np.array([function(el) for el in result_es_mu]))
            best_points_es_mu[i].append(point_es_mu)

            print(f"Iter: {i}, Seed: {seed_value} finished")

        curves_de[i] = np.add.reduce(results_de) / SEED_COUNT
        curves_es_mu[i] = np.add.reduce(results_es_mu) / SEED_COUNT

        calls_de[i] = [pop_size + 2 * pop_size *
                       j for j in range(max_iter_de + 1)]
        calls_es_mu[i] = [mu + (mu + lambd) * (j + 1)
                          for j in range(max_iter_es_mu)]


if __name__ == "__main__":
    best_de = []
    best_es_mu = []
    params = []

    functions = [ackley, sphere, poly, disturbed_square, himmelblau, eggholder]
    function_dict = {}

    for function in functions:
        function_dict[function.__name__] = function

    with open("experiment_results/best_hyper_de.csv", "r") as file:
        for line in file.readlines():
            name, sigma1, sigma2, dimensions, pop_size, diff_weight, cross_prob, max_iters, _, _, _, _ = line.split(
                ",")
            best_de.append((int(pop_size), float(
                diff_weight), float(cross_prob), int(max_iters)))
            params.append((function_dict[name], float(
                sigma1), float(sigma2), int(dimensions)))

    with open("experiment_results/best_hyper_mu_plus_lambda.csv", "r") as file:
        for line in file.readlines():
            _, _, _, _, mu, lambd, init_mut_strength, max_iters, _, _, _, _ = line.split(
                ",")
            best_es_mu.append(
                (int(mu), int(lambd), float(init_mut_strength), int(max_iters)))

    with Manager() as manager:
        curves_de = manager.list([None for _ in params])
        curves_es_mu = manager.list([None for _ in params])

        calls_de = manager.list([None for _ in params])
        calls_es_mu = manager.list([None for _ in params])

        best_points_de = manager.list([manager.list([]) for _ in params])
        best_points_es_mu = manager.list([manager.list([]) for _ in params])

        with Pool(processes=5) as pool:
            pool.starmap(get_plot_data, [(best_de, best_es_mu, params, curves_de, curves_es_mu,
                                          calls_de, calls_es_mu, best_points_de, best_points_es_mu, 12 * i, 12 * (i + 1)) for i in range(5)])

        curves_de = list(curves_de)
        curves_es_mu = list(curves_es_mu)
        calls_de = list(calls_de)
        calls_es_mu = list(calls_es_mu)
        best_points_de = list(best_points_de)
        for i in range(len(best_points_de)):
            best_points_de[i] = list(best_points_de[i])
        best_points_es_mu = list(best_points_es_mu)
        for i in range(len(best_points_es_mu)):
            best_points_es_mu[i] = list(best_points_es_mu[i])

    with open("plot_data.pkl", "wb") as file:
        pickle.dump([params, curves_de, curves_es_mu, calls_de,
                    calls_es_mu, best_points_de, best_points_es_mu], file)
