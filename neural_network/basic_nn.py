import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

train_set = pd.read_csv('./dados/optdigits.tra', header = None)
test_set = pd.read_csv('./dados/optdigits.tes', header = None)

raw_data = train_set.values[:, :-1]
classes = train_set.values[:, 64]

test_data = test_set.values[:, :-1]
excepted_classes = test_set.values[:, 64]

max_iters = [100, 1000, 10000]
layer_sizes = [(37,), (74,), (148,)]
learning_rates = [0.1, 0.01, 0.001]

nn_classifier = MLPClassifier(momentum=0.8, max_iter=100, hidden_layer_sizes=(10,), learning_rate_init=0.01)

nn_classifier = nn_classifier.fit(raw_data, classes)

predict_results = nn_classifier.predict(test_data)

correct = 0
wrong = 0
for row in range(len(predict_results)):
    if predict_results[row] == excepted_classes[row]:
        correct = correct + 1
    else:
        wrong = wrong + 1

print "NN results:"
print "Correct:", correct, ", incorrect:", wrong