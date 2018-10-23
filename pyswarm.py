from pyswarm import pso
import pandas as pd
import numpy as np

train_set = pd.read_csv('./dados/optdigits.tra', header = None)
test_set = pd.read_csv('./dados/optdigits.tes', header = None)

base_corr = train_set.corr()[64]

def fitness(ind):
    normalized = [1 if x >= 0.5 else 0 for x in ind]
    selected_columns = np.where(np.array(normalized) == 1)[0]
    final_selection = [base_corr[i] for i in selected_columns]
    return final_selection.mean()

lb = [0 for _ in range(64)]
ub = [1 for _ in range(64)]

xopt, fopt = pso(fitness, lb, ub)

print xopt
print fopt