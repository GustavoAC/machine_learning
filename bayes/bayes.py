import pandas as pd
import numpy as np
from scipy import stats
from sklearn.naive_bayes import GaussianNB

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

train_set = pd.read_csv('../dados/optdigits.tra', header = None)
test_set = pd.read_csv('../dados/optdigits.tes', header = None)

#10-fold cross-validation
k_fold = 10

complete_data = np.concatenate((train_set.values, test_set.values), axis=0)
normal = 0
not_normal = 0
for i in range(64):
    attribs = complete_data[:,i]
    k2, p = stats.normaltest(attribs)
    alpha = 1e-3
    if p < alpha:
        not_normal += 1
    else:
        normal += 1

print "Normal", normal
print "Not normal", not_normal

print ">>> Gauss Naive Bayes"
gnb = GaussianNB()
k_fold_cross_validation(k_fold, complete_data, gnb)