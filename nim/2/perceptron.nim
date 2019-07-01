import ../utils
import sequtils

type Perceptron = ref object
    layer: seq[seq[Perceptron]]
    w: array[2, float]
    b: float


proc call(p: Perceptron, x1: bool, x2: bool): bool =
    if p.layer.len == 0:
        # single layer
        var sum: float = 0.0
        let x = [float(x1), float(x2)]
        for item in zip(p.w, x):
            sum += item[0] * item[1]

        if sum+p.b > 0:
            result = true
        else:
            result = false
    else:
        result = p.layer[0][0].call(p.layer[1][0].call(x1, x2), p.layer[1][1].call(x1, x2))

const table = @[@[false, false],
                @[false, true],
                @[true, false],
                @[true, true]]

let AND: Perceptron = Perceptron(w: [0.5, 0.5], b: -0.7)
let NAND: Perceptron = Perceptron(w: [-0.5, -0.5], b: 0.7)
let OR: Perceptron = Perceptron(w: [1.0, 1.0], b: -0.5)
let XOR: Perceptron = Perceptron(layer: @[@[AND], @[NAND, OR]])
let and_table = table.map do (x: seq[bool]) -> bool: AND.call(x[0], x[1])
let nand_table = table.map do (x: seq[bool]) -> bool: NAND.call(x[0], x[1])
let or_table = table.map do (x: seq[bool]) -> bool: OR.call(x[0], x[1])
let xor_table = table.map do (x: seq[bool]) -> bool: XOR.call(x[0], x[1])
echo and_table
echo nand_table
echo or_table
echo xor_table
