'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for digit in range(10):
        digit_data = data.get_digits_by_label(train_data, train_labels, digit)
        means[digit] = np.mean(digit_data, axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for digit in range(10):
        digit_data = data.get_digits_by_label(train_data, train_labels, digit)
        diff = digit_data - means[digit]
        stabilizer = 0.01 * np.identity(diff.shape[1])
        dot_prod = np.matmul(diff.transpose(), diff)
        covariances[digit] = dot_prod/diff.shape[0] + stabilizer
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    log_matrx = np.zeros((10, 8, 8))
    for digit in range(10):
        cov_diagonal = np.diagonal(covariances[digit])
        log_diagonal = np.log(cov_diagonal)
        log_matrx[digit] = log_diagonal.reshape(8,8)

    # Plot all log-diagonal of each covariance on same axis
    all_concat = np.concatenate(log_matrx, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    gen_l = np.zeros((digits.shape[0], 10))
    d = digits.shape[1]
    for digit in range(10):
        det = np.linalg.det(covariances[digit])
        inverse = np.linalg.inv(covariances[digit])
        for x in range(digits.shape[0]):
            diff = (digits[x]-means[digit]).reshape(1, d)
            log_p = (-d/2) * np.log(2 * np.pi) - (1/2) * np.log(det) - (1/2) * (np.matmul(np.matmul(diff, inverse), diff.transpose()))
            gen_l[x, digit] = log_p
    return gen_l

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_l = generative_likelihood(digits, means, covariances)
    denominator = np.sum(np.exp(gen_l) * 0.1, axis=1).reshape(-1, 1)
    return gen_l + np.log(0.1) - np.log(denominator)


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_l = conditional_likelihood(digits, means, covariances)
    # Compute as described above and return
    num_data = digits.shape[0]
    total = 0
    for x in range(digits.shape[0]):
        correct_label = int(labels[x])
        total += cond_l[x, correct_label]
    return total / num_data

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def accuracy(predictions, labels):
    '''
    Return the accuracy of our predictions
    '''
    correct = 0
    num_labs = len(labels)
    for i in range(num_labs):
        if predictions[i] == int(labels[i]):
            correct += 1
    return correct / num_labs

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    
    # Evaluation
    # Plot the graph
    plot_cov_diagonal(covariances)

    avg_cond_l_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_cond_l_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("Avg conditional log likelihood (Train):", avg_cond_l_train)
    print("Avg conditional log likelihood (Test):", avg_cond_l_test)


    class_train = classify_data(train_data, means, covariances)
    class_test = classify_data(test_data, means, covariances)

    print("Accuracy (Train):", accuracy(class_train, train_labels))
    print("Accuracy (Test):", accuracy(class_test, test_labels))

if __name__ == '__main__':
    main()