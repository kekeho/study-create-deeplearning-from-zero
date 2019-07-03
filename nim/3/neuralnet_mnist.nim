import json
import neo
import tables
import os
import sequtils
import sugar
import ../dataset/mnist
import ./triple_neuralnet


proc get_data(): array[2, Matrix[float64]] =
    let data = load_mnist(normalize=true, flatten=true, one_hot_label=false)
    return [data["test_img"], data["test_label"]]

proc init_network(): Network =
    let pwd = currentSourcePath.splitPath.head
    let json_obj = json.parseFile(os.joinPath(pwd, "sample_weight.json"))

    let w_list = @[json_obj["W1"], json_obj["W2"], json_obj["W3"]].map(x => x.to(seq[seq[float64]]))
    let b_list = @[json_obj["b1"], json_obj["b2"],json_obj["b3"]].map(x => x.to(seq[float64]))

    var net = Network()
    net.weight_list = w_list.map(x => matrix(x))
    net.bias_list = b_list.map(x => matrix(@[x]))
    net.h = sigmoid
    net.sigma = softmax
    return net

if isMainModule:
    let data = get_data()
    let x: Matrix[float64] = data[0]
    let t: Vector[float64] = data[1].asVector
    let net = init_network()

    var accuracy_cnt = 0
    for i in 0..x.M-1:
        let y = net.forward(x[i..i, All])
        let p = y.asVector.toSeq.find(y.max)  # index

        if p == int(t[i]):
            accuracy_cnt += 1
    
    echo "Done!"
    echo "Accuracy: ", (accuracy_cnt / x.M) * 100 , '%'
