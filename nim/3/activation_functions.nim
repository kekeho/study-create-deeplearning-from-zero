import neo
import ../utils
import plotly

proc sigmoid(x: Vector): auto=
    result = 1 /! (1 +! exp(x *! -1))


let x = arange(-5, 5, 0.1, dtype=float)
let y = sigmoid(x)

let data = Trace[float](mode: PlotMode.Lines)
data.xs = x.toSeq
data.ys = y.toSeq

let p = Plot[float](traces: @[data])
p.show()
