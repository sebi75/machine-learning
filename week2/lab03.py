import numpy as np
import matplotlib.pyplot as plt
from main import run_gradient_descent, cost_function

np.set_printoptions(precision=2)
plt.style.use("./datascience.mplstyle")


def zscore_normalize_features(x_train):
    """
    computes  X, zcore normalized by column

    Args:
      x_train (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(x_train, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(x_train, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    x_norm = (x_train - mu) / sigma

    return x_norm, mu, sigma


def plot_cost_i_w(X, y, hist):
    ws = np.array([p[0] for p in hist["params"]])
    rng = max(abs(ws[:,0].min()),abs(ws[:,0].max()))
    wr = np.linspace(-rng+0.27,rng+0.27,20)
    cst = [cost_function(X,y,np.array([wr[i],-32, -67, -1.46]), 221) for i in range(len(wr))]

    fig, ax = plt.subplots(1,2,figsize=(12, 3))
    ax[0].plot(hist["iter"], (hist["cost"]));  ax[0].set_title("Cost vs Iteration")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("Cost")
    ax[1].plot(wr, cst); ax[1].set_title("Cost vs w[0]")
    ax[1].set_xlabel("w[0]"); ax[1].set_ylabel("Cost")
    ax[1].plot(ws[:, 0], hist["cost"])
    plt.show()


def load_house_data():
    data = np.loadtxt("./houses.txt", delimiter=',', skiprows=1)
    x_train = data[:, :4]
    y_train = data[:, 4]
    return x_train, y_train


# plot the features to see how each feature affects the price
def plot_data_intuition(x_train, y_train, x_features):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x_train[:, i], y_train)
        ax[i].set_xlabel(x_features[i])
    ax[0].set_ylabel("Price (100s)")
    plt.show()


# dataset will be with columns: size (sqft), number of bedrooms, number of floors, age of home, price
def main():
    x_train, y_train = load_house_data()
    x_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

    # normalize the original features
    x_norm, x_mu, x_sigma = zscore_normalize_features(x_train)

    plot_data_intuition(x_norm, y_train, x_features)

    alpha = 1.0e-1
    _, _, hist = run_gradient_descent(x_train, y_train, 10, alpha)
    plot_cost_i_w(x_norm, y_train, hist)


if __name__ == "__main__":
    main()
