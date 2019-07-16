from types import FunctionType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numerical_diff_1d import numerical_diff
import numpy as np
from types import FunctionType


def function_2(x, y: int or float) -> int or float:
    return x**2 + y**2


def numerial_gradient(f: FunctionType, x: int or float) -> float:
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(*x)

        x[idx] = tmp_val - h
        fxh2 = f(*x)

        grad[idx] = (fxh1 - fxh2) / 2 * h
        x[idx] = tmp_val

    return grad

function_tmp1 = lambda x: function_2(x, 4.0)
function_tmp2 = lambda y: function_2(3.0, y)


def main() -> None:
    print(numerical_diff(function_tmp1, 3.0))
    print(numerical_diff(function_tmp2, 4.0))

    print(numerial_gradient(function_2, np.array([3.0, 4.0])))

    x_range = np.arange(-3, 3, 0.1)
    y_range = np.arange(-3, 3, 0.1)
    z = [[function_2(x, y) for y in y_range] for x in x_range]

    X, Y = map(lambda x: x.flatten(), np.meshgrid(x_range, y_range))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X, Y, z)
    plt.show()


if __name__ == "__main__":
    main()
