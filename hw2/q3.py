from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

def computer_a_i(test_datum, x_train, tau):
    # Find the weight of the test_datum
    test_datum_transpose = np.reshape(test_datum, (1, d))
    val = -l2(test_datum_transpose, x_train) / (2*(tau**2))
    return np.exp(val - logsumexp(val))

#to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Given a test datum, it returns its prediction based on locally weighted regression

    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    a_i = computer_a_i(test_datum, x_train, tau)
    test_datum_transpose = np.reshape(test_datum, (1, test_datum.shape[0]))
    A = np.diag(np.reshape(a_i, (a_i.shape[1],)))
    x_transpose = np.transpose(x_train)
    matrix_A = np.matmul(np.matmul(x_transpose, A), x_train)
    matrix_B = np.matmul(np.matmul(x_transpose, A), y_train)
    w = np.linalg.solve(matrix_A + 1e-8 * np.eye(matrix_A.shape[0]), matrix_B)
    return np.matmul(test_datum_transpose, w)
    ## TODO

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses

#to implement
def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    # split into folds
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    losses = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        losses.append(run_on_fold(x_test, y_test, x_train, y_train, taus))
    ## TODO
    losses_sum = sum(losses[i] for i in range(k)) # This sums 200 diff tau values up and return a list of length 5
    avg_losses = losses_sum / k
    return avg_losses
    ## TODO


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    # Plotting 
    plt.title("Tau to Avg Losses")
    plt.xlabel("Tau val")
    plt.ylabel("Avg Losses")
    plt.plot(taus, losses)
    plt.show()

    print("min loss = {}".format(losses.min()))

