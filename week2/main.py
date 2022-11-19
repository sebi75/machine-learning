import numpy as np
import math
import matplotlib.pyplot as plt


def gradient_descent(x_train, y_train, w_initial, b_initial, cost_fn, gradient_compute, number_of_iterations, alpha):
    # in cost functions list we will save all the costs obtained in the iterations
    cost_functions_list = []
    w_and_b_pairs = []
    w = w_initial
    b = b_initial

    for i in range(number_of_iterations):
        # compute the partial derivatives in each iteration for updating the parameters
        dj_dw, dj_db = gradient_compute(x_train, y_train, w, b)

        # update the parameters simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # compute the cost with the new parameters and save it
        cost_functions_list.append(cost_fn(x_train, y_train, w, b))

        # save the new pairs
        w_and_b_pairs.append([w, b])

        # show progress through iterations only at 1000 iterations
        if i % math.ceil(number_of_iterations / 10) == 0:
            print(f"Iteration: {i}: cost: {cost_functions_list[-1]} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, cost_functions_list, w_and_b_pairs


def compute_gradient(x_train, y_train, w, b):
    # initialize the derivatives with zero, to compute in them
    dj_dw = 0
    dj_db = 0
    m = len(x_train)
    # iterate from 0 to m - 1 to use all the training examples
    for i in range(m):
        # use the formulas for the partial derivatives to iteratively compute
        f_wb = np.dot(x_train[i], w) + b
        dj_dw += (f_wb - y_train[i]) * x_train[i]
        dj_db += (f_wb - y_train[i])

    return dj_dw / m, dj_db / m


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

    return cost / (2 * m)


# we want to be able to plot data without prediction too, initialize it with None
def plot_data(x_train, y_train, tmp_f_wb=None):
    if tmp_f_wb is not None:
        plt.plot(x_train, tmp_f_wb, marker='x', c='b', label='Our prediction')

    # plot the points
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    plt.title("Housing prices")
    plt.show()


def calculate_f_wb(inp, w, b):
    return np.dot(w, inp) + b


def main():
    # prices are in $1000's
    y_train = np.array([100, 234, 567, 345, 123, 976, 456, 123, 264, 476, 654, 345])
    # now the x_train will have multiple features: land print (in 100s sqm), no of bedrooms,
    # no of bathrooms, year of construction

    # each row represents the features for one training example
    # cols will be in order: land print, noOfBedrooms, noOfBathrooms, yearOfConstruction
    x_train = np.array([[4, 2, 1, 2000],
                       [6, 3, 2, 2005],
                       [10, 5, 3, 2010],
                       [7, 4, 2, 2005],
                       [3, 2, 1, 1995],
                       [15, 5, 3, 2015],
                       [8, 3, 2, 2018],
                       [5, 3, 1, 1985],
                       [6, 3, 2, 1980],
                       [10, 3, 2, 1998],
                       [15, 4, 2, 2015],
                       [7, 4, 2, 2005],
    ])

    # run the model
    w, b, cost_functions_list, w_and_b_pairs = gradient_descent(x_train, y_train, 0, 0, cost_function, compute_gradient, 10000, 0.1)

    print(f"costs history: {cost_functions_list}",
          f"w and b pairs: {w_and_b_pairs}")

    sample_input_for_prediction = np.array([7, 4, 2, 2005])
    print(f"prediction for input: {sample_input_for_prediction} is:",
          f"{calculate_f_wb(sample_input_for_prediction, w, b)}")


if __name__ == "__main__":
    main()