import layer_naive

apple = 100
apple_count = 2
orange = 150
orange_count = 3
tax = 1.1

mul_apple_layer = layer_naive.MulLayer()
mul_orange_layer = layer_naive.MulLayer()

add_apple_orange_layer = layer_naive.AddLayer()
mul_tax_layer = layer_naive.MulLayer()

# forward
mul_apple = mul_apple_layer.forward(apple, apple_count)
mul_orange = mul_orange_layer.forward(orange, orange_count)

add_apple_orange = add_apple_orange_layer.forward(mul_apple, mul_orange)

total_price = mul_tax_layer.forward(add_apple_orange, tax)
print(total_price)


# backward
dtotal_price = 1
dadd_apple_orange, dtax = mul_tax_layer.backward(dtotal_price)

dmul_apple, dmul_orange = add_apple_orange_layer.backward(dadd_apple_orange)

dapple, dapple_count = mul_apple_layer.backward(dmul_apple)
dorange, dorange_count = mul_orange_layer.backward(dmul_orange)

print(dapple_count, dapple, dorange, dorange_count, dtax)

