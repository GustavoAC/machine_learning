import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def k_fold_cross_validation(k_fold, complete_data, algorithm):
    start_index = 0
    end_index = 0
    total_correct = 0
    total_wrong = 0
    for i in range(k_fold):
        start_index = len(complete_data) / k_fold * i
        end_index = len(complete_data) / k_fold * (i + 1)
        i_fold_test = complete_data[ start_index : end_index ]
        i_fold_train = list(complete_data)
        del i_fold_train[ start_index : end_index ]
        i_fold_train = np.array(i_fold_train)
            
        train_data = i_fold_train[:, :-1]
        train_classes = i_fold_train[:, 64]
        
        test_data = i_fold_test[:, :-1]
        test_classes = i_fold_test[:, 64]
        
        classifier = algorithm
        classifier = classifier.fit(train_data, train_classes)
        predict_results = classifier.predict(test_data)
        
        correct = 0
        wrong = 0
        for row in range(len(predict_results)):
            if predict_results[row] == test_classes[row]:
                correct = correct + 1
            else:
                wrong = wrong + 1
        total_correct += correct
        total_wrong += wrong
    print "Correct: ", total_correct
    print "Wrong: ", total_wrong
    print "Total: ", len(complete_data)    

train_set = pd.read_csv('./dados/optdigits.tra', header = None)
test_set = pd.read_csv('./dados/optdigits.tes', header = None)

#10-fold cross-validation
k_fold = 10

complete_data = np.concatenate((train_set.values, test_set.values), axis=0)

## Uniform (no-weight)
for k in [1, 3, 5, 7, 9]:
    print k, "NN (uniform):"
    knn_classifier = KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
    k_fold_cross_validation(k_fold, complete_data, knn_classifier)

## Weighted (no-weight)
for k in [1, 3, 5, 7, 9]:
    print k, "NN (weighted):"
    knn_classifier = KNeighborsClassifier(n_neighbors = k, weights = 'distance')
    k_fold_cross_validation(k_fold, complete_data, knn_classifier)