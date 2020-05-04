from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np 
import data

from metric_report import *


def svm_classifier(train_data, train_labels):
    # Create/train and search for optimal parameters for a SVM classifier
    parameters = {'kernel':('linear', 'poly', 'rbf'), 'C':[10, 100, 1000]} # Add more params
    svc = SVC(decision_function_shape='ovo')
    SVM = GridSearchCV(svc, parameters)
    SVM.fit(train_data, train_labels)
    return SVM

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    #SVM Classifier
    svm = svm_classifier(train_data, train_labels)
    predictions = svm.predict(test_data)

    # summarize results
    print("Best: %f using %s" % (svm.best_score_, svm.best_params_))
    means = svm.cv_results_['mean_test_score']
    stds = svm.cv_results_['std_test_score']
    params = svm.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    report(test_labels, predictions)
    