import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train_set = pd.read_csv('./dados/optdigits.tra', header = None)
test_set = pd.read_csv('./dados/optdigits.tes', header = None)

# print train_set.corr()[64].mean()

raw_data = train_set.values[:, :-1]
classes = train_set.values[:, 64]

test_data = test_set.values[:, :-1]
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
