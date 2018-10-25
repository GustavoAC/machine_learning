import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train_set = pd.read_csv('./dados/optdigits.tra', header = None)
test_set = pd.read_csv('./dados/optdigits.tes', header = None)

base_corr = train_set.corr()[64]
filter = [1 if np.abs(corr) > 0.1 else 0 for corr in base_corr]
print "Filter:", filter

def filter_set(data, filter):
    ret = []
    for row in data:
        new_row = [row[i] for i in range(len(row)) if filter[i] == 1]
        ret.append(new_row)

    return ret

raw_data = filter_set(train_set.values[:, :-1], filter)
classes = train_set.values[:, 64]

test_data = filter_set(test_set.values[:, :-1], filter)
excepted_classes = test_set.values[:, 64]

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

