import neo
import math
import sequtils


proc mean_squared_error(y: Matrix[float64], teacher: Matrix[float64]): float64 =
    return neo.sum((y - teacher).map(proc(x: float64): float64 = x ^ 2)) / 2

proc cross_entropy_error(y: Matrix[float64], teacher: Matrix[float64]): float64 =
    let delta = 1e-7
    let y_add_delta: Matrix[float64] = y + neo.constantMatrix(y.M, y.N, delta)
    let logged = y_add_delta.map(proc (x: float64): float64 = log(x, exp(1.0)))

    return -1.0 * sum(teacher |*| logged)

if isMainModule:
    let t = neo.matrix(@[@[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    let y1 = neo.matrix(@[@[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]])
    let y2 = neo.matrix(@[@[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]])

    echo "Mean squared error:"
    echo mean_squared_error(y1, t)
    echo mean_squared_error(y2, t)

    echo "Cross entrypy error:"
    echo cross_entropy_error(y1, t)
    echo cross_entropy_error(y2, t)
