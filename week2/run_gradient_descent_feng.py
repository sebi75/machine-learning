import numpy as np
from main import gradient_descent, cost_function
from compute_gradient_matrix import compute_gradient_matrix


def run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-6):
    m, n = X.shape
    # initialize parameters
    initial_w = np.zeros(n)
    initial_b = 0
    # run gradient descent
    w_out, b_out, hist_out = gradient_descent(X, y, initial_w, initial_b,
                                              cost_function, compute_gradient_matrix, alpha, iterations)
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.4f}")

    return w_out, b_out
