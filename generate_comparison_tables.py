from functions import *
from scipy.stats import ttest_ind
import pickle

with open("plot_data.pkl", "rb+") as file:
    params, curves_de, curves_es_mu, calls_de, calls_es_mu, best_points_de, best_points_es_mu = pickle.load(
        file)

student_comparison = [None for _ in params]

for i, param in enumerate(params):
    de_values = [param[0](point) for point in best_points_de[i]]
    es_mu_values = [param[0](point) for point in best_points_es_mu[i]]
    test_result = ttest_ind(de_values, es_mu_values, equal_var=False)

    if test_result.pvalue >= 0.1 or test_result.statistic == 0:
        student_comparison[i] = "--"
    elif test_result.statistic < 0:
        student_comparison[i] = "DE"
    elif test_result.statistic > 0:
        student_comparison[i] = "ES"

comp_11 = []
comp_110 = []
comp_101 = []
comp_1010 = []
for comp, param in zip(student_comparison, params):
    if param[1] == 1:
        if param[2] == 1:
            comp_11.append(comp)
        elif param[2] == 10:
            comp_110.append(comp)
    elif param[1] == 10:
        if param[2] == 1:
            comp_101.append(comp)
        elif param[2] == 10:
            comp_1010.append(comp)


def group_comp(comp):
    new_comp = []
    for i in range(4):
        row = [comp[3 * i], comp[3 * i + 1], comp[3 * i + 2]]
        new_comp.append(row)
    for i in range(2):
        new_comp.append(comp[i + 12])

    return new_comp


def draw_latex_table(comp):
    comp = group_comp(comp)

    comp3d = comp[:4]
    comp1d = comp[4:]
    functions3d = ["ackley", "sphere", "poly", "disturbed square"]
    functions1d = ["himmelblau", "eggholder"]

    print(r"\begin{tabular}{|c|c|c|c|}")
    print(r"    \hline")
    print(r"    Nazwa funkcji \textbackslash Wymiarowość& 10& 30& 50 \\")
    for i, name in enumerate(functions3d):
        print(r"    \hline")
        print(
            fr"    {name}& {comp3d[i][0]}& {comp3d[i][1]}& {comp3d[i][2]} \\")
    print(r"    \hline")
    print(r"\end{tabular}")
    print(r"\quad")
    print(r"\begin{tabular}{|c|c|}")
    print(r"    \hline")
    print(r"    Nazwa funkcji \textbackslash Wymiarowość& 2 \\")
    for i, name in enumerate(functions1d):
        print(r"    \hline")
        print(fr"    {name}& {comp1d[i]} \\")
    print(r"    \hline")
    print(r"\end{tabular}")
    print("")


draw_latex_table(comp_11)
draw_latex_table(comp_110)
draw_latex_table(comp_101)
draw_latex_table(comp_1010)
