from pyeasyga import pyeasyga
import random
import pandas as pd
import numpy as np

train_set = pd.read_csv('./dados/optdigits.tra', header = None)
test_set = pd.read_csv('./dados/optdigits.tes', header = None)

base_corr = train_set.corr()[64]

def create_random_ind(data):
    return [random.randint(0, 1) for _ in range(len(train_set.columns) - 1)]

# define a fitness function
def fitness(individual, data):
    selected_columns = np.where(np.array(individual) == 1)[0]
    final_selection = [np.abs(data[i]) for i in selected_columns]
 
    if (final_selection == []): return 0
 
    res = np.asarray(final_selection).mean()
    if (np.isnan(res)):
        return 0
    return res

print "Starting... "
ga = pyeasyga.GeneticAlgorithm(base_corr)
ga.create_individual = create_random_ind
ga.fitness_function = fitness
ga.run()
print ga.best_individual()