from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.scatter(X[:, i], y, s=5)
        plt.title(features[i])
        plt.xlabel("Feature Value")
        plt.ylabel("Response")
    
    plt.tight_layout()
    plt.show()

def normalize_feature(feature_vector):
    # Normalize a feature vector
    mean = sum(feature_vector) / len(feature_vector)
    sd = (sum([((feature_vector[i] - mean)**2) for i in range(len(feature_vector))]) / len(feature_vector))**(1/2)
    new_array = np.array([((feature_vector[i] - mean) / sd) for i in range(len(feature_vector))])
    return np.reshape(new_array, (len(feature_vector), 1)) 

def fit_regression(X, y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    
    X_transpose_X = np.matmul(np.transpose(X), X)
    X_transpose_y = np.matmul(np.transpose(X), y)
    w = np.linalg.solve(X_transpose_X , X_transpose_y) 
    return w

def predict(X, w):
    return np.matmul(X, w)

def mean_percentage_error(predict_vals, target):
    return sum(((predict_vals[i] - target[i])/predict_vals[i]) for i in range(len(target))) / len(target)

def mean_squared_error(predict_vals, target):
    return sum([(predict_vals[i] - target[i])**2 for i in range(len(target))]) / len(target)

def mean_absolute_percentage_error(predict_vals, target):
    return sum(abs((predict_vals[i] - target[i])/predict_vals[i]) for i in range(len(target))) / len(target)

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    print("Number of data points: " + str(len(y)))
    print("Dimension of Input data: " + str(X.shape))
    print("Dimension of Input target: " + str(y.shape))

    # Normalize X
    if len(features) > 0:
        new_X = normalize_feature(X[:,0])
    for i in range(len(features) - 1):
        new_feature_vector = normalize_feature(X[:,i + 1])
        new_X = np.concatenate((new_X, new_feature_vector), axis=1)
    
    # Visualize the features
    visualize(new_X, y, features)

    matrix_ones = np.ones((X.shape[0], 1), dtype=int)
    new_X = np.concatenate((matrix_ones, new_X), axis=1)
    #TODO: Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.25, random_state=20)
    
    # Fit regression model
    w = fit_regression(X_train, y_train)
        
    # Print the feature | weight
    max_len = max([len(i) for i in features])
    print("Feature" + " "*(max_len - len("Feature") + 1) + "| Weight")
    for i in range(len(features)):
        print(features[i] + " "*(max_len - len(features[i]) + 1) + "|" + " " + str(w[i + 1]))

    # Compute fitted values, MSE, etc.
    predict_vals = predict(X_test, w)
    MSE = mean_squared_error(predict_vals, y_test)
    print("Mean Squared Error: " + str(MSE))
    MAPE = mean_absolute_percentage_error(predict_vals, y_test) * 100
    print("Mean Absolute Percentage Error: " + str(MAPE) + " %")
    MPE = mean_percentage_error(predict_vals, y_test) 
    print("Mean Percentage Error: " + str(MPE))

if __name__ == "__main__":
    main()

