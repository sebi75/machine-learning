import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab03 import load_house_data

dlc = dict(dlblue='#0096ff', dlorange='#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')

np.set_printoptions(precision=2)
plt.style.use('./datascience.mplstyle')


def normalize_training_data(x_train):
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x_train)

    return x_norm


def check_before_and_after_normalization(x_train, x_norm):
    print(f"Peak to Peak range by column in raw        X: {np.ptp(x_train, axis=0)}")
    print(f"Peak to Peak range by column in Normalized X: {np.ptp(x_norm, axis=0)}")


def create_and_fit_model(x_train, y_train):
    model = SGDRegressor(max_iter=1000)
    model.fit(x_train, y_train)  # the x_train is actually the normalized input

    print(model)
    print(f"number of iters: {model.n_iter_}, number of weight updates: {model.t_}")

    # view the parameters
    b_norm = model.intercept_
    w_norm = model.coef_
    print(f"model parameters:                   w: {w_norm}, b: {b_norm}")
    print(f"model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

    return model, w_norm, b_norm


def plot_results(x_train, x_features, y_train, y_pred):
    # plot predictions and targets vs original features
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x_train[:, i], y_train, label='target')
        ax[i].set_xlabel(x_features[i])
        ax[i].scatter(x_train[:, i], y_pred, color=dlc["dlorange"], label='predict')
    ax[0].set_ylabel("Price");
    ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()


def main():
    x_train, y_train = load_house_data()
    x_features = ["size(sqft)", "bedrooms", "floors", "age"]

    x_norm = normalize_training_data(x_train)

    check_before_and_after_normalization(x_train, x_norm)

    trained_model, w_norm, b_norm = create_and_fit_model(x_norm, y_train)

    prediction = trained_model.predict(x_norm)

    # make prediction using w and b
    y_pred = np.dot(x_norm, w_norm) + b_norm
    print(f"prediction using np.dot() and sgdr,predict match: {(y_pred == prediction).all()}")

    print(f"Prediction on training set:\n{y_pred[:4]}")
    print(f"Target values \n{y_train[:4]}")

    plot_results(x_train, x_features, y_train, y_pred, )


if __name__ == "__main__":
    main()
