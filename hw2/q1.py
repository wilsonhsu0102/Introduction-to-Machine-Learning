import matplotlib.pyplot as plt

def plot_bias_variance(mu, var, n, lambd_min, lambd_max):
    bias_list = []
    variance_list = []
    expected_squared_error = []
    x_axis = []
    for lambd in range(lambd_min, lambd_max):
        bias = (lambd * mu / (lambd + 1))**2
        bias_list.append(bias)
        variance = var/(n*(lambd + 1)**2)
        variance_list.append(variance)
        expected_squared_error.append(variance + bias**2)
        x_axis.append(lambd)

    plt.plot(x_axis, bias_list, x_axis, variance_list, x_axis, expected_squared_error)
    plt.xlabel('Lambda')
    plt.ylabel('Bias/Variance/Expected Squared Err')
    plt.xticks(x_axis)
    plt.title('Lambda vs Bias/Variance')
    plt.legend(('Bias', 'Variance', 'Expected Squared Error'), loc='right')
    plt.show()

if "__main__" == __name__:
    plot_bias_variance(1, 9, 10, 0, 20)
