import numpy as np
from types import FunctionType

def numerial_gradient(f: FunctionType, x: np.ndarray) -> np.ndarray:
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f: FunctionType, init_x: np.ndarray, lr=0.01, step_num=100) -> np.ndarray:
    x = init_x

    for i in range(step_num):
        grad = numerial_gradient(f, x)
        x -= (lr * grad)

    return x


def function_2(x: np.ndarray):
    return x[0] ** 2 + x[1] ** 2


def main() -> None:
    init_x = np.array([-3.0, 4.0])
    result = gradient_descent(f=function_2, init_x=init_x, lr=0.1, step_num=100)
    print(result)

if __name__ == "__main__":
    main()