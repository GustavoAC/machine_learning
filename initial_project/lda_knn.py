import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

train_set = pd.read_csv('./dados/optdigits.tra', header = None)
test_set = pd.read_csv('./dados/optdigits.tes', header = None)

# Training data and classes
train_img = train_set.values[:, :-1]
train_lbl = train_set.values[:, 64]

# Test data and classes
test_img = test_set.values[:, :-1]
test_lbl = test_set.values[:, 64]

lda = LinearDiscriminantAnalysis()
lda.fit(train_img, train_lbl)

train_img = lda.transform(train_img)
test_img = lda.transform(test_img)

print "Num Components:", len(test_img[0])

knn_classifier = KNeighborsClassifier()
knn_classifier = knn_classifier.fit(train_img, train_lbl)

predict_results = knn_classifier.predict(test_img)

correct = 0
wrong = 0
for row in range(len(predict_results)):
    if predict_results[row] == test_lbl[row]:
        correct = correct + 1
    else:
        wrong = wrong + 1

print "Correct", correct
print "Wrong", wrong
