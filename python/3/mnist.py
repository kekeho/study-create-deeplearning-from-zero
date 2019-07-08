from activation_functions import sigmoid, softmax
from triple_neuralnet import Network
import sys
import os
import numpy as np
import pickle

sys.path.append(os.path.abspath('../deep-learning-from-scratch/'))
from dataset.mnist import load_mnist


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True,
                                                      one_hot_label=False)
    return x_test, t_test


def init_network() -> Network:
    with open('../deep-learning-from-scratch/ch03/sample_weight.pkl', 'rb') as f:
        data = pickle.load(f)

    w_l = [data['W1'], data['W2'], data['W3']]
    b_l = [data['b1'], data['b2'], data['b3']]
    network = Network(w_list=w_l, b_list=b_l,
                      activation_func=sigmoid, finalize_func=softmax)
    return network


def main():
    x, t = get_data()
    network = init_network()

    accuracy_count = 0
    for i in range(len(x)):
        y = network.forward(x[i])
        p = np.argmax(y)

        if p == t[i]:
            accuracy_count += 1

    print('Done!')
    print(f'Accuracy: {accuracy_count / len(x) * 100}%')


if __name__ == "__main__":
    main()
