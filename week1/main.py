import math
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./datascience.mplstyle")


def gradient_descent(x_train, y_train, w_init, b_init, alpha, iter_count, cost_function, gradient_function):
    J_history = []
    pair_history = []
    b = b_init
    w = w_init
    for i in range(iter_count):
        dj_dw, dj_db = gradient_function(x_train, y_train, w, b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        J_history.append(cost_function(x_train, y_train, w, b))
        pair_history.append([w, b])

        if i % math.ceil(iter_count / 10) == 0:
            print(f"Iteration: {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, pair_history # return w and J, w history for graphing


def compute_gradient(x_train, y_train, w, b):
    """this function calculates the partial derivatives for w and b"""
    m = x_train.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = x_train[i] * w + b
        dj_dw += (f_wb - y_train[i]) * x_train[i]
        dj_db += (f_wb - y_train[i])
    return dj_dw / m, dj_db / m


def compute_model_output(x, w, b):
    """computes the prediction of a linear model"""
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


def plot_data(x_train, y_train, tmp_f_wb=None):
    # plot out prediction:
    if tmp_f_wb is not None:
        plt.plot(x_train, tmp_f_wb, marker='x', c='b', label='Our prediction')

    # plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual values')
    # set the title
    plt.title("Housing prices")
    # set the y-axis label
    plt.ylabel("Prices (1000s dollars)")
    # set the x-axis label
    plt.xlabel("Landsize")
    plt.legend()
    plt.show()


def cost_function(x_train, y_train, w, b):
    m = x_train.shape[0]
    cost = 0
    for i in range(m):
        f_wb = x_train[i] * w + b
        cost += (f_wb - y_train[i]) ** 2

    return (1 / (2 * m)) * cost


def main():
    x_train = np.array([1.0, 2.0, 2.5, 1.3, 2.5, 1.3, 8.7])
    y_train = np.array([100, 200, 250, 137, 255, 133, 876])

    w, b, j_history, pairs = gradient_descent(x_train, y_train, 0, 0, 0.05, 10000, cost_function, compute_gradient)

    print(f"last w and b: w:{w}, b:{b}")
    print(f"history of costs obtained: {j_history}")
    tmp_w, tmp_b = pairs[35]
    tmp_fwb = compute_model_output(x_train, tmp_w, tmp_b)

    plot_data(x_train, y_train, tmp_fwb)

    print(f"prediction for x: 20: {w * 20 + b}")


if __name__ == '__main__':
    main()

