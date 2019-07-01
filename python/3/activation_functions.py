import numpy as np
from matplotlib import pyplot as plt


def step(x: np.ndarray or int or float):
    y = x > 0
    return y


def sigmoid(x: np.ndarray or int or float):
    y = 1 / (1 + np.exp(-x))
    return y


def relu(x: np.ndarray or int or float):
    return np.maximum(0, x)


def main():
    x = np.arange(-5, 5, 0.1)
    y_step = step(x)
    y_sigmoid = sigmoid(x)
    y_relu = relu(x)
    plt.plot(x, y_step)
    plt.plot(x, y_sigmoid)
    plt.plot(x, y_relu)
    plt.show()


if __name__ == "__main__":
    main()
