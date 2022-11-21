import math
import numpy as np
from statistics import mean


def calculate_f_wb(inp, w, b):
    return np.dot(w, inp) + b


def calculate_accuracy(x_train, y_train, w, b):
    # we use this function to calculate the MSE
    errors = []
    for i in range(x_train.shape[0]):
        prediction = calculate_f_wb(x_train[i], w, b)
        errors.append((prediction - y_train[i]) ** 2)

    mse = math.sqrt(mean(errors))
    return f"MSE: {mse} (in 1000s of dollars)"