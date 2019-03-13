import neo
import typetraits
import sequtils

iterator zip*(a: any, b: any): array =
    # Check seqs length
    if a.len != b.len:
        var e = new IndexError
        e.msg = "Each seqs must be the same length."
        raise e
        
    for i in 0..a.len-1:
        yield [a[i], b[i]]


proc shapeArray*(x: Matrix): array[2, int] =
    result = [x.M, x.N]


proc arange*(start, stop, step: int or float = 1, dtype=int): auto =
    let
        start = start.dtype
        stop = stop.dtype
        step = step.dtype

    var tmp: seq[dtype] = @[start]
    var add_value = 0.dtype
    while true:
        add_value += step
        if start + add_value > stop:
            break

        tmp.insert((start + add_value), tmp.len)
    
    result = vector(tmp)


proc toseq*[T](vec: Vector[T]): seq[T] =
    var ret: seq[T] = @[]
    for i in vec:
        ret.insert(i, ret.len)
    
    result = ret


proc `+!`*(vec: Vector[int], value: int): Vector[int] =
    var ret_int: seq[int] = @[]

    # generate int vector
    for x in vec:
        ret_int.insert(x.int + value.int, ret_int.len)

    return vector(ret_int)


proc `+!`*(vec: Vector, value: int or float): Vector[float] =
    var ret_float: seq[float] = @[]

    # generate float vector
    for x in vec:
        ret_float.insert(x.float + value.float, ret_float.len)

    return vector(ret_float)


proc `+!`*(value: int, vec: Vector[int]): Vector[int] =
    var ret_int: seq[int] = @[]

    # generate int vector
    for x in vec:
        ret_int.insert(x.int + value.int, ret_int.len)

    return vector(ret_int)


proc `+!`*(value: int or float, vec: Vector): Vector[float] =
    var ret_float: seq[float] = @[]

    # generate int vector
    for x in vec:
        ret_float.insert(x.float + value.float, ret_float.len)

    return vector(ret_float)


proc `-!`*(vec: Vector[int], value: int): Vector[int] =
    var ret_int: seq[int] = @[]

    # generate int vector
    for x in vec:
        ret_int.insert(x.int - value.int, ret_int.len)

    return vector(ret_int)


proc `-!`*(vec: Vector, value: int or float): Vector[float] =
    var ret_float: seq[float] = @[]

    # generate float vector
    for x in vec:
        ret_float.insert(x.float - value.float, ret_float.len)

    return vector(ret_float)


proc `-!`*(value: int, vec: Vector[int]): Vector[int] =
    var ret_int: seq[int] = @[]

    # generate int vector
    for x in vec:
        ret_int.insert(value.int - x.int, ret_int.len)

    return vector(ret_int)


proc `-!`*(value: int or float, vec: Vector): Vector[float] =
    var ret_float: seq[float] = @[]

    # generate float vector
    for x in vec:
        ret_float.insert(value.float - x.float, ret_float.len)

    return vector(ret_float)


proc `*!`*(vec: Vector[int], value: int): Vector[int] =
    var ret_int: seq[int] = @[]

    # generate int vector
    for x in vec:
        ret_int.insert(x.int * value.int, ret_int.len)

    return vector(ret_int)


proc `*!`*(vec: Vector, value: int or float): Vector[float] =
    var ret_float: seq[float] = @[]

    # generate float vector
    for x in vec:
        ret_float.insert(x.float * value.float, ret_float.len)

    return vector(ret_float)


proc `*!`*(value: int, vec: Vector[int]): Vector[int] =
    var ret_int: seq[int] = @[]

    # generate int vector
    for x in vec:
        ret_int.insert(value.int * x.int, ret_int.len)

    return vector(ret_int)


proc `*!`*(value: int or float, vec: Vector): Vector[float] =
    var ret_float: seq[float] = @[]

    # generate float vector
    for x in vec:
        ret_float.insert(value.float * x.float, ret_float.len)

    return vector(ret_float)


proc `/!`*(vec: Vector, value: int or float): Vector[float] =
    var ret: seq[float] = @[]

    # generate float vector
    for x in vec:
        ret.insert(x.float / value.float, ret.len)

    return vector(ret)


proc `/!`*(value: int or float, vec: Vector): Vector[float] =
    var ret: seq[float] = @[]

    # generate int vector
    for x in vec:
        ret.insert(value.float / x.float, ret.len)

    return vector(ret)


proc main(): void =
    # let x = @[0, 1, 2, 3, 4]
    # let y = [5, 6, 7, 8, 9]

    # for i in zip(x, y):
    #     echo i[0], ' ', i[1]
    
    let a = arange(0.0, 1.0, 0.5, float)    # [ 0.0, 0.5, 1.0 ]
    let b = arange(0, 5, 2)  # [ 0, 2, 4 ]
    let c = arange(0, 5, 2, float)  # [ 0.0, 2.0, 4.0 ]
    let d = arange(0, 5, dtype=float)  # [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ]

    echo b +! 1    # [ 1     3       5 ]
    echo 1 +! b    # [ 1     3       5 ]
    echo b +! 1.0  # [ 1.0   3.0     5.0 ]
    echo 1.0 +! b  # [ 1.0   3.0     5.0 ]
    echo c +! 1    # [ 1.0   3.0     5.0 ]
    echo 1 +! c    # [ 1.0   3.0     5.0 ]
    echo c +! 1.0  # [ 1.0   3.0     5.0 ]
    echo 1.0 +! c  # [ 1.0   3.0     5.0 ]
    echo c /! 2    # [ 0.0   1.0     2.0 ]
    echo 2 /! b    # [ inf   1.0     0.5 ]



if isMainModule:
    main()