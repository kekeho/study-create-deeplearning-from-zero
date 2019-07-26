import layer_naive

apple = 100
apple_count = 2
tax = 1.1

# layer
mul_apple_layer = layer_naive.MulLayer()
mul_tax_layer = layer_naive.MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_count)
price_with_tax = mul_tax_layer.forward(apple_price, tax)

print(price_with_tax)


# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dcount = mul_apple_layer.backward(dapple_price)

print(dapple, dcount, dtax)

