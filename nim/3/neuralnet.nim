import neo

let X = neo.matrix(@[@[1.0, 2.0]])
let W = neo.matrix(@[@[1.0, 3.0, 5.0],
                     @[2.0, 4.0, 6.0]])

let Y = X * W

echo Y
