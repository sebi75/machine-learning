import copy
import numpy as np
import math
import matplotlib.pyplot as plt
from cost_function import cost_function
from calculate_accuracy import calculate_f_wb, calculate_accuracy

np.set_printoptions(precision=2)


def compute_gradient(x_train, y_train, w, b):
    # initialize the derivatives with zero, to compute in them
    m, n = x_train.shape
    dj_db = 0
    dj_dw = np.zeros((n,))
    # iterate from 0 to m - 1 to use all the training examples
    for i in range(m):
        # use the formulas for the partial derivatives to iteratively compute
        error = calculate_f_wb(x_train[i], w, b) - y_train[i]
        for j in range(n):
            dj_dw[j] += error * x_train[i, j]
        dj_db += error

    return dj_dw / m, dj_db / m


# def gradient_descent(x_train, y_train, number_of_iterations, alpha):
#     # in cost functions list we will save all the costs obtained in the iterations
#     m, n = x_train.shape
#     w = np.zeros(n)
#     b = 0
#
#     for i in range(number_of_iterations):
#         # compute the partial derivatives in each iteration for updating the parameters
#         dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)
#
#         # update the parameters simultaneously
#         w = w - alpha * dj_dw
#         b = b - alpha * dj_db
#
#         # compute the cost with the new parameters and save it
#         cost_functions_list.append(cost_function(x_train, y_train, w, b))
#
#         # save the new pairs
#         w_and_b_pairs.append([w, b])
#
#         # show progress through iterations only at 1000 iterations
#         if i % math.ceil(number_of_iterations / 10) == 0:
#             print(f"Iteration: {i}: cost: {cost_functions_list[-1]}")
#
#     return w, b, cost_functions_list, w_and_b_pairs


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X : (array_like Shape (m,n)    matrix of examples
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(X)

    # An array to store values at each iteration primarily for graphing later
    hist = {}
    hist["cost"] = []; hist["params"] = []; hist["grads"] = []; hist["iter"] = [];

    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    save_interval = np.ceil(num_iters / 10000)  # prevent resource exhaustion for long runs

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J,w,b at each save interval for graphing
        if i == 0 or i % save_interval == 0:
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append([w, b])
            hist["grads"].append([dj_dw, dj_db])
            hist["iter"].append(i)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            # print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
            cst = cost_function(X, y, w, b)
            print(f"Iteration {i:9d}, Cost: {cst:0.5e}")
    return w, b, hist  # return w,b and history for graphing


def gradient_descent_houses(x_train, y_train, initial_w, initial_b, compute_gradient_matrix, alpha, iterations):
    """
    :param x_train: array like shape (m. n) matrix of examples
    :param y_train: array like shape (m) target value of each example
    :param initial_w: attay like shape (n,) initial values for parameters of the model
    :param initial_b: initial value of parameter of the model
    :param compute_gradient_matrix: function to compute the gradient
    :param alpha: the learning rate
    :param iterations: number of iterations to apply the gradient
    :return: w: array like shape (n,) updated values of the parameters of the model after running the gradient
            b: (scalar) updated bias after running the gradient descent
    """
    m = len(x_train)  # number of training examples
    # an array for storing values at each iteration for graphing later
    hist = {}
    hist["cost"] = []; hist["params"] = []; hist["grads"] = []; hist["iter"] = [];

    w = copy.deepcopy(initial_w)  # deep copy so we don't edit the global w
    b = initial_b
    save_interval = np.ceil(iterations / 10000)  # prevent resource exhaustion for long runs

    print(
        f"Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  ")
    print(
        f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|")

    for i in range(iterations):

        # calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient_matrix(x_train, y_train, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # save the cost J, w, b at each interval
        if i == 0 or i % save_interval == 0:
            hist["cost"].append(cost_function(x_train, y_train, w, b))
            hist["params"].append([w, b])
            hist["grads"].append([dj_dw, dj_db])
            hist["iter"].append(i)

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(iterations / 10) == 0:
                # print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
                cst = cost_function(x_train, y_train, w, b)
                print(
                    f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")

    return w, b, hist


def run_gradient_descent(x_train, y_train, iterations=1000, alpha=1e-6):
    m, n = x_train.shape
    # initialize parameters
    initial_w = np.zeros(n)
    initial_b = 0
    # run gradient descent
    w_out, b_out, hist_out = gradient_descent_houses(x_train, y_train, initial_w, initial_b, compute_gradient, alpha,
                                                     iterations)
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")

    return w_out, b_out, hist_out


# we want to be able to plot data without prediction too, initialize it with None
def plot_data(x_train, y_train, tmp_f_wb=None):
    if tmp_f_wb is not None:
        plt.plot(x_train, tmp_f_wb, marker='x', c='b', label='Our prediction')

    # plot the points
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    plt.title("Housing prices")
    plt.show()


def main():
    # prices are in $1000's
    y_train = np.array([100, 234, 567, 345, 123, 976, 456, 123, 264, 476, 654, 345])
    # now the x_train will have multiple features: land print (in 100s sqm), no of bedrooms,
    # no of bathrooms, ageOfHome

    # each row represents the features for one training example
    # cols will be in order: land print, noOfBedrooms, noOfBathrooms, ageOfHome
    x_train = np.array([[4, 2, 1, 22],
                       [6, 3, 2, 17],
                       [10, 5, 3, 12],
                       [7, 4, 2, 17],
                       [3, 2, 1, 27],
                       [15, 5, 3, 7],
                       [8, 3, 2, 3],
                       [5, 3, 1, 40],
                       [6, 3, 2, 42],
                       [10, 3, 2, 24],
                       [15, 4, 2, 7],
                       [7, 4, 2, 17]])

    w_init = np.zeros((x_train.shape[1],))
    b_init = 0
    alpha = 5.0e-7
    num_of_iterations = 200000

    # run the model
    w, b, cost_functions_list, w_and_b_pairs = gradient_descent_houses(x_train, y_train, w_init, b_init, compute_gradient, alpha, num_of_iterations)

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(cost_functions_list)
    ax2.plot(100 + np.arange(len(cost_functions_list[100000:])), cost_functions_list[100000:])
    ax1.set_title("Cost vs. iteration");
    ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost');
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step');
    ax2.set_xlabel('iteration step')
    plt.show()

    print(f"costs history: {cost_functions_list}")

    sample_input_for_prediction = np.array([7, 4, 2, 17])
    print(f"prediction for input: {sample_input_for_prediction} is:",
          f"{calculate_f_wb(sample_input_for_prediction, w, b)}")

    accuracy_message = calculate_accuracy(x_train, y_train, w, b)
    print(accuracy_message)


if __name__ == "__main__":
    main()
