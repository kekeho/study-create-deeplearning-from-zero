import numpy as np

def mean_squared_error(y: np.ndarray, teacher: np.ndarray) -> float:
    return np.sum((y - teacher) ** 2) / 2


def cross_entropy_error(y: np.ndarray, teacher: np.ndarray) -> float:
    delta = 1e-7
    return -1 * np.sum(t * np.log(y + delta))


if __name__ == "__main__":
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])  # 2
    y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])  # 6
    print("Mean squared error:")
    print(mean_squared_error(y1, t))
    print(mean_squared_error(y2, t))

    print("Cross entropy error:")
    print(cross_entropy_error(y1, t))
    print(cross_entropy_error(y2, t))