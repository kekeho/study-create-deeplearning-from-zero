import matplotlib.pyplot as plt
import numpy as np
from types import FunctionType, LambdaType


def numerical_diff(f: FunctionType, x: int or float) -> float:
    h = 1e-4  # delta
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x: int or float) -> float:
    return 0.01*x**2 + 0.1*x


def gradient(f: FunctionType, x: int or float) -> LambdaType:
    diff = numerical_diff(f, x)
    height = f(x) - diff * x
    return lambda x: diff * x + height


def main() -> None:
    x = np.arange(0, 20)
    y = list(map(function_1, x))
    plt.plot(x, y)

    [plt.plot(x, list(map(gradient(function_1, val), x))) for val in x]
    plt.show()


if __name__ == "__main__":
    main()
