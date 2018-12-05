import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

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

complete_data = np.concatenate((train_set.values, test_set.values), axis=0)

max_iter = 100
layer_size = (37,)
learning_rate = 0.1

for i in range(5):
    print "Execution #", i + 1
    nn_classifier = MLPClassifier(momentum=0.8, max_iter=max_iter, hidden_layer_sizes=layer_size, learning_rate_init=learning_rate)
    k_fold_cross_validation(10, complete_data, nn_classifier)