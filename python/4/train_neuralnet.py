import numpy as np
import os
import sys
sys.path.append(os.path.join(os.pardir, 'deep-learning-from-scratch/dataset'))
from mnist import load_mnist
from two_layer_net import TwoLayerNet
from matplotlib import pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

# ハイパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

print('Initializing network...')
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
print('Start learning...')

for i in range(iters_num):
    print(f'\r{i} of {iters_num}', end='')
    # get minibatch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # cal grad
    grad = network.numerial_gradient(x_batch, t_batch)

    # update params
    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key]
    
    # recoginize learning
    loss = network.loss(x_batch, t_batch)
    plt.scatter(i, loss, c="blue")
    plt.pause(0.001)
    train_loss_list.append(loss)

# for x, y in enumerate(train_loss_list):
    # plt.plot(x, y)

# plt.show()

