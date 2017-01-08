from .net import convVars, conv2d, flatten, fullVars, full, pool2d, dropout


def Lenet(x, output_depth, do_dropout=True):
    """ Create a LeNet network
        x inputs should be 32x32xX
    """
    # Hyperparameters

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    input_depth = int(x.get_shape()[-1])
    print("input_depth:", input_depth)
    filter_size = 5
    W_1, b_1 = convVars(filter_size, filter_size, input_depth, 6, '1')
    net = conv2d(x, W_1, b_1, strides=1, padding='VALID')
    print("L1", net)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    net = pool2d(net)
    print("P1", net)

    # Layer 2: Convolutional. Output = 10x10x16.
    filter_size = 5
    W_2, b_2 = convVars(filter_size, filter_size, 6, 16, '2')
    net = conv2d(net, W_2, b_2, strides=1, padding='VALID')
    print("L2", net)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    net = pool2d(net)
    print("P2", net)

    # Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(net)

    print("flat:", flat)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    W_3, b_3 = fullVars(400, 120, 'f3')
    net = full(flat, W_3, b_3)
    print("Lf3", net)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    W_4, b_4 = fullVars(120, 84, 'f4')
    net = full(net, W_4, b_4)
    print("Lf4", net)

    if do_dropout:
        net = dropout(net)
        print("Dropout", net)

    # Layer 5: Fully Connected. Input = 84. Output = <output_depth>.
    W_5, b_5 = fullVars(84, output_depth, 'f5')
    logits = full(net, W_5, b_5)
    print("out", logits)

    return logits
