import os
import sys
import numpy as np
from loss_function import cross_entropy_error
sys.path.append(os.path.join(os.pardir, '3'))
from activation_functions import softmax
from numerical_diff_2d import numerial_gradient


class SimpleNet:
    def __init__(self, *args, **kwargs):
        self.W = np.random.randn(2, 3)
    
    
    def predict(self, x):
        return np.dot(x, self.W)
    

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


def main() -> None:
    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))

    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    f = lambda dammy: net.loss(x, t)
    dW = numerial_gradient(f, net.W)
    print(dW)


if __name__ == "__main__":
    main()