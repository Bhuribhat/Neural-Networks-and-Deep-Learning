import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model


def generate_data():
    np.random.seed(0)
    x, y = datasets.make_moons(200, noise=0.20)
    return x, y


def visualize(x, y, clf):
    plot_decision_boundary(lambda x: clf.predict(x), x, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, x, y):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def classify(x, y):
    clf = linear_model.LogisticRegressionCV()
    clf.fit(x, y)
    return clf


def main():
    x, y = generate_data()
    clf = classify(x, y)
    visualize(x, y, clf)


if __name__ == "__main__":
    main()