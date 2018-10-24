import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

train_set = pd.read_csv('./dados/optdigits.tra', header = None)
test_set = pd.read_csv('./dados/optdigits.tes', header = None)

# Training data and classes
train_img = train_set.values[:, :-1]
train_lbl = train_set.values[:, 64]

# Test data and classes
test_img = test_set.values[:, :-1]
test_lbl = test_set.values[:, 64]

# scaler = StandardScaler()

# # Fit on training set only.
# scaler.fit(train_img)

# # Apply transform to both the training set and the test set.
# train_img = scaler.transform(train_img)
# test_img = scaler.transform(test_img)

pca = LinearDiscriminantAnalysis()
pca.fit(train_img, train_lbl)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

decisionTree = DecisionTreeClassifier()
decisionTree = decisionTree.fit(train_img, train_lbl)

predict_results = decisionTree.predict(test_img)

correct = 0
wrong = 0
for row in range(len(predict_results)):
    if predict_results[row] == test_lbl[row]:
        correct = correct + 1
    else:
        wrong = wrong + 1

print "Correct", correct
print "Wrong", wrong
