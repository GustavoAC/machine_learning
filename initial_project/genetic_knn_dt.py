from pyeasyga import pyeasyga
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

print "Starting genetic... "
ga = pyeasyga.GeneticAlgorithm(base_corr,
                               population_size=10,
                               generations=20,
                               crossover_probability=0.8,
                               mutation_probability=0.1,
                               elitism=True,
                               maximise_fitness=True)

ga.create_individual = create_random_ind
ga.fitness_function = fitness
ga.run()

print "Genetic algorithm best individual:", ga.best_individual()

######### Classifiers below

def filter_set(data, filter):
    ret = []
    for row in data:
        new_row = [row[i] for i in range(len(row)) if filter[i] == 1]
        ret.append(new_row)

    return ret

raw_data = train_set.values[:, :-1]
classes = train_set.values[:, 64]

test_data = test_set.values[:, :-1]
excepted_classes = test_set.values[:, 64]

raw_data = filter_set(raw_data, ga.best_individual()[1])
test_data = filter_set(test_data, ga.best_individual()[1])

# KNN

knn_classifier = KNeighborsClassifier()
knn_classifier = knn_classifier.fit(raw_data, classes)

predict_results = knn_classifier.predict(test_data)

correct = 0
wrong = 0
for row in range(len(predict_results)):
    if predict_results[row] == excepted_classes[row]:
        correct = correct + 1
    else:
        wrong = wrong + 1

print "KNN results:"
print "Correct:", correct, ", incorrect:", wrong
print ""

# Decision Tree

decisionTree = DecisionTreeClassifier()
decisionTree = decisionTree.fit(raw_data, classes)

predict_results = decisionTree.predict(test_data)

correct = 0
wrong = 0
for row in range(len(predict_results)):
    if predict_results[row] == excepted_classes[row]:
        correct = correct + 1
    else:
        wrong = wrong + 1

print "Decision tree results:"
print "Correct:", correct, ", incorrect:", wrong
