import tables
import os
import httpclient
import osproc
import tables
import neo
import sugar
import sequtils


let url_base = "http://yann.lecun.com/exdb/mnist/"
let key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz"
}.toTable

let dataset_dir = system.currentSourcePath.splitFile.dir
let save_file = "mnist_param.json"


let train_num = 60000
let test_num = 10000
let img_dim = (1, 28, 28)
let img_size = 784


proc download(filename: string): void =
    let file_path = os.joinPath(dataset_dir, filename)
    let uncomprewssed_filename = os.joinPath(dataset_dir, filename.splitFile.name)
    if not os.existsFile(uncomprewssed_filename):
        echo "Downloading ", filename, " ..."
        let client = newHttpClient()
        client.downloadFile(url_base & filename, file_path)


proc download_mnist(): void = 
        for k, v in key_file:
            download(v)


proc load_label(filename: string): Matrix[float64] =
    let file_path = os.joinPath(dataset_dir, filename)
    echo "Loading label from ", filename, " ..."

    let uncompressed_filename = os.joinPath(file_path.splitFile.dir, file_path.splitFile.name)
    if not os.existsFile(uncompressed_filename):
        # Uncompress
        let exit_code = osproc.execCmd("gzip -df " & file_path)
        if exit_code != 0:
            quit(-1)

    # Read labels
    let raw_filename = os.joinPath(file_path.splitFile.dir, file_path.splitFile.name)
    let rawfile = open(raw_filename, FileMode.fmRead)
    var buffer: seq[uint8]
    buffer.newSeq(rawfile.getFileSize)
    let cnt = rawfile.readBytes(buffer, 0, rawfile.getFileSize)
    buffer = buffer[8..buffer.len-1]

    let float_buffer = buffer.map(x => float64(x))
    return neo.matrix(@[float_buffer])


proc load_img(filename: string): Matrix[float64] =
    let file_path = os.joinPath(dataset_dir, filename)
    let uncompressed_filename = os.joinPath(file_path.splitFile.dir, file_path.splitFile.name)

    echo "Loading img from ", filename, " ..."

    if not existsFile(uncompressed_filename):
        # Uncompress
        echo "Unzip ", file_path, " to ", uncompressed_filename
        let exit_code = osproc.execCmd("gzip -df " & file_path)
        if exit_code != 0:
            quit(-1)

    # Read imgs
    let raw_filename = os.joinPath(file_path.splitFile.dir, file_path.splitFile.name)
    let rawfile = open(raw_filename, FileMode.fmRead)
    var buffer: seq[uint8]
    buffer.newSeq(rawfile.getFileSize)
    let cnt = rawfile.readBytes(buffer, 0, rawfile.getFileSize)
    buffer = buffer[16..buffer.len-1]

    var img_seq = newSeq[seq[float64]]()
    for i in countup(0, buffer.len-1, img_size):
        if buffer.len > i + img_size:
            img_seq.add(buffer[i..(i + img_size - 1)].map(x => float64(x)))

    return neo.matrix(img_seq)

proc convert_seq(): Table[string, Matrix[float64]] =
    var dataset = initTable[string, Matrix[float64]]()
    dataset["train_img"] = load_img(key_file["train_img"])
    dataset["train_label"] = load_label(key_file["train_label"])
    dataset["test_img"] = load_img(key_file["test_img"])
    dataset["test_label"] = load_label(key_file["test_label"])

    return dataset

proc init_mnist(): void =
    download_mnist()
    echo "Finish initialize"


proc change_one_hot_label(x: Matrix[float64]): Matrix[float64] =
    var x_vector = x.asVector
    var return_seq = newSeq[seq[float64]]()
    for i in x_vector:
        var tmp = neo.constantVector(10, 0.0)
        tmp[int(i)] = 1.0
        return_seq.add(tmp.toSeq)
    
    return matrix(return_seq)



proc load_mnist*(normalize=true, flatten=true, one_hot_label=false): Table[string, Matrix[float64]] =
    init_mnist()
    var dataset: Table[string, Matrix[float64]] = convert_seq()

    if normalize:
        for key in ["train_img", "test_img"]:
            dataset[key] /= 255.0
    
    if one_hot_label:
        for key in ["train_label", "test_label"]:
            dataset[key] = change_one_hot_label(dataset[key])

    # if not flatten:
        # TODO: あとで書く
    
    return dataset