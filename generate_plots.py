import numpy as np
from functions import *
import matplotlib.pyplot as plt
import pickle

with open("plot_data.pkl", "rb+") as file:
    params, curves_de, curves_es_mu, calls_de, calls_es_mu, best_points_de, best_points_es_mu = pickle.load(
        file)

eps = 1e-2

for i, param in enumerate(params):
    function, sigma1, sigma2, dimensions = param

    max_both = max(max(curves_de[i]), max(curves_es_mu[i]))
    min_both = min(min(curves_de[i]), min(curves_es_mu[i]))

    plt.plot(calls_de[i], np.log10(curves_de[i] - min_both + eps))
    plt.plot(calls_es_mu[i], np.log10(curves_es_mu[i] - min_both + eps))
    yticks = plt.yticks()[0]
    ylabels = np.power(10, yticks) + min_both - eps
    ylabels = [f"{label:.1e}" if len(
        f"{label:.2f}") > 7 else f"{label:6.3f}" for label in ylabels]
    plt.yticks(yticks, ylabels)
    plt.title(
        f"Krzywe zbieżności dla funkcji {function.__name__} {dimensions}-wymiarowej\n" +
        f"przy zaburzeniach ({sigma1},{sigma2})")
    plt.xlabel("Liczba obliczeń funkcji celu")
    plt.ylabel("")
    plt.legend(["DE", "ES μ + λ"])
    plt.savefig(
        f"plots/conv_{function.__name__}_{sigma1}_{sigma2}_{dimensions}.png")
    plt.clf()

    barriers = np.geomspace(max_both - min_both, eps, 100) + min_both
    ecdf_de = np.array([sum(value < barriers)
                       for value in curves_de[i]]) / len(barriers)
    ecdf_es_mu = np.array([sum(value < barriers)
                          for value in curves_es_mu[i]]) / len(barriers)
    plt.plot(calls_de[i], ecdf_de)
    plt.plot(calls_es_mu[i], ecdf_es_mu)
    plt.title(
        f"Krzywe ECDF dla funkcji {function.__name__} {dimensions}-wymiarowej\n" +
        f"przy zaburzeniach ({sigma1},{sigma2})")
    plt.xlabel("Liczba obliczeń funkcji celu")
    plt.ylabel("Względna liczba przekroczonych barier")
    plt.legend(["DE", "ES μ + λ"])
    plt.savefig(
        f"plots/ecdf_{function.__name__}_{sigma1}_{sigma2}_{dimensions}.png")
    plt.clf()

params_3d = []
for i, param in enumerate(params):
    function, sigma1, sigma2, dimensions = param
    if function not in [eggholder, himmelblau]:
        params_3d.append(i)

i = 0
while i < len(params):
    function, sigma1, sigma2, dimensions = params[i]
    if i in params_3d:
        de_colors = ["blue", "mediumblue", "darkblue"]
        es_mu_colors = ["orange", "darkorange", "orangered"]
        for j in range(3):
            max_both = max(*curves_de[i + j], *curves_es_mu[i + j])
            min_both = min(*curves_de[i + j], *curves_es_mu[i + j])
            barriers = np.geomspace(max_both - min_both, eps, 100) + min_both
            ecdf_de = np.array([sum(value < barriers)
                               for value in curves_de[i + j]]) / len(barriers)
            ecdf_es_mu = np.array([sum(value < barriers)
                                  for value in curves_es_mu[i + j]]) / len(barriers)
            plt.plot(calls_de[i + j], ecdf_de, color=de_colors[j])
            plt.plot(calls_es_mu[i + j], ecdf_es_mu, color=es_mu_colors[j])
        plt.title(
            f"Krzywe ECDF dla funkcji {function.__name__}\n" +
            f"przy zaburzeniach ({sigma1},{sigma2})")
        plt.xlabel("Liczba obliczeń funkcji celu")
        plt.ylabel("Względna liczba przekroczonych barier")
        plt.legend(["DE/10", "ES μ + λ/10", "DE/30",
                   "ES μ + λ/30", "DE/50", "ES μ + λ/50"])
        plt.savefig(
            f"plots/ecdf_{function.__name__}_{sigma1}_{sigma2}_combined.png")
        plt.clf()
        i += 3
    else:
        i += 2
