import numpy as np


def cost_function(x_train, y_train, w, b):
    # for multiple input data we need to use dot product for the computation
    # f_wb formula:
    # one feature: f_wb = w * x_train[i] + b
    # multiple features: f_wb = np.dot(x_train[i], w) + b, x_train[i] and w are vectors
    # formula: 1/2m * (sum(i) 0 -> m - 1 ( f_wb - y_train[i] ) ** 2)
    m = len(x_train)
    cost = 0
    for i in range(m):
        f_wb = np.dot(x_train[i], w) + b
        cost += (f_wb - y_train[i]) ** 2

    cost = cost / (2 * m)
    return np.squeeze(cost)