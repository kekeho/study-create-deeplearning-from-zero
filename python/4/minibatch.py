import sys
import os
import numpy as np

sys.path.append(os.path.abspath('../deep-learning-from-scratch/'))
from dataset.mnist import load_mnist


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    print(t_batch)

if __name__ == "__main__":
    main()