import numpy as np


class Perceptron():
    def __init__(self, w=[], b=0.0, layer=None):
        self.layer = layer
        self.w = np.array(w)
        self.b = b

    def __multi_layer_call(self, x1: int or float, x2: int or float, f_list):
        """再帰処理用"""
        try:
            # 再帰的にパーセプトロンを呼び出し
            return f_list[0].call(
                    self.__multi_layer_call(x1, x2, f_list=f_list[1][0]),
                    self.__multi_layer_call(x1, x2, f_list=f_list[1][1])
                )
        except TypeError:
            # 末尾
            return f_list.call(x1, x2)

    def call(self, x1, x2) -> bool:
        if not self.layer:
            # Single layer
            x = np.array([x1, x2])

            if np.sum(self.w * x) + self.b <= 0:
                return False
            else:
                # 発火
                return True

        else:
            # Multi layer
            return self.__multi_layer_call(x1, x2, self.layer)


if __name__ == "__main__":
    AND = Perceptron(w=[0.5, 0.5], b=-0.7)
    NAND = Perceptron(w=[-0.5, -0.5], b=0.7)
    OR = Perceptron(w=[1, 1], b=-0.5)
    XOR = Perceptron(layer=[AND, [NAND, OR]])

    xor_table = [
        XOR.call(0, 0),
        XOR.call(0, 1),
        XOR.call(1, 0),
        XOR.call(1, 1),
    ]
    print(xor_table)
