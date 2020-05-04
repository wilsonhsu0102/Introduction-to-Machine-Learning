'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

a = 2
b = 2
def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    for digit in range(10):
        digit_data = data.get_digits_by_label(train_data, train_labels, digit)
        total = digit_data.shape[0]
        theta = (np.sum(digit_data, axis=0) + a - 1)/(total + a + b - 2)
        eta[digit] = theta
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    images = []
    for i in range(10):
        img_i = class_images[i]
        images.append(img_i.reshape(8, 8))
    all_concat = np.concatenate(images, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for k in range(10):
        generated_data[k] = [np.random.choice([0, 1], 1, p=[1 - eta[k, j], eta[k, j]]) for j in range(eta.shape[1])]
    plot_images(np.array(generated_data))

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    gen_l = np.zeros((bin_digits.shape[0], 10))
    for j in range(bin_digits.shape[0]):
        gen_l[j] = [np.sum(bin_digits[j] * np.log(eta[k]) + (1 - bin_digits[j]) * np.log(1 - eta[k])) for k in range(10)]
    return np.array(gen_l)

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_l = generative_likelihood(bin_digits, eta)
    denominator = np.sum(np.exp(gen_l) * 0.1, axis=1).reshape(-1, 1)
    return gen_l + np.log(0.1) - np.log(denominator)

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_l = conditional_likelihood(bin_digits, eta)
    # Compute as described above and return
    num_data = bin_digits.shape[0]
    total = 0
    for x in range(bin_digits.shape[0]):
        correct_label = int(labels[x])
        total += cond_l[x, correct_label]
    return total / num_data

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
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
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)
    avg_cond_l_train = avg_conditional_likelihood(train_data, train_labels, eta)
    avg_cond_l_test = avg_conditional_likelihood(test_data, test_labels, eta)
    print("Avg conditional log likelihood (Train):", avg_cond_l_train)
    print("Avg conditional log likelihood (Test):", avg_cond_l_test)

    class_train = classify_data(train_data, eta)
    class_test = classify_data(test_data, eta)
    print("Accuracy (Train):", accuracy(class_train, train_labels))
    print("Accuracy (Test):", accuracy(class_test, test_labels))

if __name__ == '__main__':
    main()
