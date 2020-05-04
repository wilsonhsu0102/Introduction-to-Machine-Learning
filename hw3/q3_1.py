'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from scipy.special import logsumexp
import tensorflow as tf

from metric_report import *

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        digit = None
        dist = self.l2_distance(test_point)
        # Returns the index of k smallest neighbors
        nearest_neighbors_labels = self.train_labels[np.argpartition(dist, k)[:k]]
        digits, counts = np.unique(nearest_neighbors_labels, return_counts=True)
        # Return the max number (Tie breaking: First occurence)
        digit = digits[np.argmax(counts)]
        return digit

    def predict_prob(self, test_point, k):
        dist = self.l2_distance(test_point)
        # Returns the index of k smallest neighbors
        nearest_neighbors_labels = self.train_labels[np.argpartition(dist, k)[:k]]
        digits, counts = np.unique(nearest_neighbors_labels, return_counts=True)
        prob = [counts[np.where(digits == i)]/k if i in digits else 0 for i in range(10)]
        return prob


def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    avg_accuracy = []
    for k in k_range:
        # Loop over folds
        accuracies = []
        for train_index, test_index in kf.split(train_data):
            x_train, x_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            
        # Evaluate k-NN
            knn = KNearestNeighbor(x_train, y_train)
            accuracies.append(classification_accuracy(knn, k, x_test, y_test))
        # Calculate average accuracy
        avg_accuracy.append(sum(accuracies) / len(accuracies))
    optimal_avg = max(avg_accuracy)
    return k_range[avg_accuracy.index(optimal_avg)], optimal_avg


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    num_data = len(eval_data)
    assert len(eval_data) == len(eval_labels)
    if num_data == 0:
        raise Exception("No data given")
    predicted_label = np.array([knn.query_knn(eval_data[i], k) for i in range(num_data)])
    correct_count = sum([1 for i in range(num_data) if predicted_label[i] == eval_labels[i]])
    score = correct_count / num_data
    return score

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    train_knn1 = classification_accuracy(knn, 1, train_data, train_labels)
    train_knn15 = classification_accuracy(knn, 15, train_data, train_labels)
    test_knn1 = classification_accuracy(knn, 1, test_data, test_labels)
    test_knn15 = classification_accuracy(knn, 15, test_data, test_labels)

    print("knn with k = 1 on Train Set has accurary:", train_knn1)
    print("knn with k = 1 on Test Set has accurary:", test_knn1)
    print("knn with k = 15 on Train Set has accurary:", train_knn15)
    print("knn with k = 15 on Test Set has accurary:", test_knn15)

    optimal_k, avg_acc = cross_validation(train_data, train_labels)
    optimal_knn_train = classification_accuracy(knn, optimal_k, train_data, train_labels)
    optimal_knn_test = classification_accuracy(knn, optimal_k, test_data, test_labels)

    print("Optimal k:", optimal_k)
    print("Optimal k average accuracy across folds:", avg_acc)
    print("Optimal knn on Train Set has accurary:", optimal_knn_train)
    print("Optimal knn on Test Set has accurary:", optimal_knn_test)

    predictions = np.array([knn.query_knn(test_data[i], optimal_k) for i in range(len(test_data))])

    report(test_labels, predictions)


if __name__ == '__main__':
    main()