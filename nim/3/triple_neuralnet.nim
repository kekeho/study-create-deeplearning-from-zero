import neo
import math
import ../utils

proc sigmoid(x: float64): float64 =
    return 1 / (1 + math.exp(-1 * x))


proc identity_function[T](x: T): T = 
    return x


type Network = ref object
    weight_list: seq[neo.Matrix[float64]]
    bias_list: seq[neo.Matrix[float64]]
    h: proc(x: float64): float64
    sigma: proc(x: float64): float64

proc depth(self: Network): int =
    return len(self.weight_list)


proc forward(self: Network, input: neo.Matrix[float64]): neo.Matrix[float64] =
    var neuron_input = input
    var a: neo.Matrix[float64]
    for i in 0..(self.depth - 1):
        a = (neuron_input * self.weight_list[i]) + self.bias_list[i]

        if i < self.depth - 1:
            neuron_input = a.map(self.h)


    return a.map(self.sigma)

let x = neo.matrix(@[@[1.0, 0.5]])
let w1 = neo.matrix(@[@[0.1, 0.3, 0.5],
                      @[0.2, 0.4, 0.6]])
let b1 = neo.matrix(@[@[0.1, 0.2, 0.3]])

let w2 = neo.matrix(@[@[0.1, 0.4],
                      @[0.2, 0.5],
                      @[0.3, 0.6]])
let b2 = neo.matrix(@[@[0.1, 0.2]])

let w3 = neo.matrix(@[@[0.1, 0.3],
                      @[0.2, 0.4]])
let b3 = neo.matrix(@[@[0.1, 0.2]])

let weight_list = @[w1, w2, w3]
let bias_list = @[b1, b2, b3]
let net = Network(weight_list: weight_list, bias_list: bias_list, h: sigmoid, sigma: identity_function)

echo net.forward(x)
