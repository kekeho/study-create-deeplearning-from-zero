import numpy as np
import activation_functions
from types import FunctionType

class Network():
    def __init__(self, w_list: np.ndarray, b_list: np.ndarray, activation_func: FunctionType, finalize_func: FunctionType):
        """Initialize network
        Args:
            w_list: Weights list
            b_list: Bias list
            activation_func: Activation Function h
            finalize_func: Finalize Function σ
        """

        self.weight_list = w_list  # Weights list
        self.bias_list = b_list  # Bias list
        self.depth = len(self.weight_list)  # Network depth
        self.h = activation_func
        self.sigma = finalize_func


    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward network
        Arg:
            x: Input
        Return:
            y: Output
        """
        neuron_input = x
        for i, (w, b) in enumerate(zip(self.weight_list, self.bias_list)):
            a = np.dot(neuron_input, w) + b

            if i == self.depth - 1:
                y = self.sigma(a)  # output
                return y

            neuron_input = self.h(a)



X = np.array([1.0, 0.5])

W1 = np.array([[0.1, 0.3, 0.5],
               [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

W2 = np.array([[0.1, 0.4],
               [0.2, 0.5],
               [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

W3 = np.array([[0.1, 0.3],
               [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

w_list = [W1, W2, W3]
b_list = [B1, B2, B3]

neuralnet = Network(w_list, b_list, activation_functions.sigmoid, activation_functions.identity_function)
y = neuralnet.forward(X)
print(y)
