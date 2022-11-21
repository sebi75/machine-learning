import numpy as np
import matplotlib.pyplot as plt
from run_gradient_descent_feng import run_gradient_descent_feng
from zscore_normalization import zscore_normalize_features

# this lab is dedicated to feature engineering


def v1():
    np.set_printoptions(precision=2)  # reduces the precision fon numpy arrays

    x = np.arange(0, 20, 1)  # initiate an array from 0 to 20 - 1 elements going with step 1
    print(x)
    X = np.c_[x, x ** 2, x ** 3]
    print(X)

    # z_score normalize features:
    X = zscore_normalize_features(X)
    print(X)

    y_train = 1 + x ** 2
    # print(f"y+train: {y_train}")

    # print(f"before feature engineering: {x}")

    x_train = x ** 2
    # print(f"after feature engineering: {x_train}")

    # x_train should be 2d matrix
    x_train = x_train.reshape(-1, 1)

    alpha = 1e-5
    # print(f"alpha is: {alpha}", )

    model_w, model_b = run_gradient_descent_feng(x_train, y_train, iterations=1000, alpha=alpha)
    plt.title("added x**2 features")
    plt.scatter(x, y_train, marker='x', c='r', label='Actual Value')
    plt.plot(x, np.dot(x_train, model_w) + model_b, label="Predicted values");
    plt.xlabel("X");
    plt.ylabel("y")
    plt.legend()
    # plt.show()


def v2():
    x = np.arange(0, 20, 1)
    y = np.cos(x / 2)

    X = np.c_[x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9, x ** 10, x ** 11, x ** 12, x ** 13]
    print(f"unnormalized: {X}")

    X = zscore_normalize_features(X)
    print(f"normalized: {X}")

    model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha=1e-1)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value")
    plt.title("Normalized x x**2, x**3 feature")
    plt.plot(x, X @ model_w + model_b, label="Predicted Value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

v2()




