import tensorflow as tf
import numpy as np 
import data

from sklearn.model_selection import GridSearchCV
from metric_report import *

def neural_net_model(dropout_rate, epochs, batch_size):
    # Creates a Neural Network model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(64,)),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        # Drop out layer to prevent overfitting
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = 'adam', metrics=['accuracy'])
    return model 

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    # Search for the best parameters. (Items removed from param list to increase run time)
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=neural_net_model)
    parameters = {'dropout_rate': [0.2], 'batch_size': [30], 'epochs': [80]} 
    grid = GridSearchCV(estimator=model, param_grid=parameters, verbose=0)
    grid_result = grid.fit(train_data, train_labels, verbose=0)

    predictions = grid_result.predict(test_data)
    report(test_labels, predictions)
