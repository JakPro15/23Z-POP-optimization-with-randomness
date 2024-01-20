import pandas as pd
from typing import TextIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_plots(data: pd.DataFrame, function: str, type: str):
    eps = 1e-2
    data["average_score"] = np.log10(
        data["average"] - min(data["average"]) + eps)
    for plot_kind in [sns.stripplot, sns.boxplot]:
        for column in data.columns.drop(["function_name", "sigma1", "sigma2", "average", "average_score"]):
            plot_kind(x=column, y="average_score", data=data)
            yticks = plt.yticks()[0]
            ylabels = np.power(10, yticks) + min(data["average"]) - eps
            ylabels = [f"{label:.1e}" for label in ylabels]
            plt.yticks(yticks, ylabels)
            plt.ylabel("")
            if type == "de":
                title_type = "ewolucji różnicowej"
            elif type == "es":
                title_type = "strategii ewolucyjnej μ + λ"
            plt.title(
                f"Wpływ hiperparametru {column} na średnią wartość\nfunkcji {function} dla algorythmu {title_type}")
            plt.savefig(
                f"./plots/{type}_{function}_{column}_{plot_kind.__name__}.png")
            plt.close()


def group_data(data: pd.DataFrame):
    grouped_data = data.groupby(list(data.drop(
        columns=["max_iterations", "average", "std_deviation", "min_score", "max_score"]).columns)).agg({
            "average": "mean"
        }).reset_index()

    return grouped_data


def save_best_rows(data: pd.DataFrame, file: TextIO):
    best_row_indexes = data.groupby(['sigma1', 'sigma2', 'dimensions'])[
        "average"].idxmin()

    for _, value in best_row_indexes.items():
        best_row = data.iloc[value].astype('str').to_list()
        file.write(','.join(best_row) + "\n")


if __name__ == "__main__":
    functions = ["ackley", "sphere", "poly",
                 "disturbed_square", "himmelblau", "eggholder"]

    with open("experiment_results/best_hyper_de.csv", "w+") as file:
        for function in functions:
            data = pd.read_csv(
                f"experiment_results/differential_evolution_{function}.csv", names=[
                    "function_name", "sigma1", "sigma2", "dimensions",
                    "population_size", "differential_weight",
                    "crossover_threshold", "max_iterations", "average", "std_deviation",
                    "min_score", "max_score"
                ])

            save_best_rows(data, file)
            grouped_data = group_data(data)
            generate_plots(grouped_data, function, "de")

    with open("experiment_results/best_hyper_mu_plus_lambda.csv", "w+") as file:
        for function in functions:
            data = pd.read_csv(
                f"experiment_results/es_mu_plus_lambda_{function}.csv", names=[
                    "function_name", "sigma1", "sigma2", "dimensions",
                    "mu", "lambda", "initial_mutation_strength", "max_iterations",
                    "average", "std_deviation",
                    "min_score", "max_score"
                ])

            save_best_rows(data, file)
            data["lambda"] = data["lambda"] / data["mu"]
            grouped_data = group_data(data)
            generate_plots(grouped_data, function, "es")
