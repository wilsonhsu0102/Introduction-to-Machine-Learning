from sklearn.ensemble import AdaBoostClassifier
import numpy as np 
import data

from metric_report import *

def digit_adaboost_classifier(digit, train_data, train_labels):
    # Create/train an adaboost classifier one vs rest for digit
    adaboost = AdaBoostClassifier(n_estimators=300)
    new_labels = [1 if label == digit else 0 for label in train_labels]
    adaboost.fit(train_data, new_labels)
    return adaboost

def digits_predict(classifiers, test_data):
    # Predict test_data based on the maximum probability out of all classifiers
    all_probs = [classifiers[i].predict_proba(test_data) for i in range(10)]
    # Get the maximum probability out of the probability that it is classified as class digit i
    predictions = [np.array(np.argmax(np.array([all_probs[digit][i][1] for digit in range(10)]))) for i in range(test_data.shape[0])]
    return np.array(predictions), all_probs

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    #Get AdaBoost Classifier for each digit
    classifiers = [digit_adaboost_classifier(i, train_data, train_labels) for i in range(10)]
    predictions, all_probs = digits_predict(classifiers, test_data)

    report(test_labels, predictions)