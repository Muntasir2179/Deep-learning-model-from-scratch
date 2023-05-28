import neuron


def activation_function(x, w):
    next_layer = [0, 0, 0]

    for i in range(0, len(w[0])):
        for j in range(0, len(x)):
            next_layer[i] += x[j] * w[j][i]

    for i in range(len(next_layer)):
        next_layer[i] = neuron.sigmoid(next_layer[i])

    return next_layer


if __name__ == '__main__':
    x = [0.8, 1]
    w1 = [[1.2, 0.7, 1], [2, 0.6, 1.8]]

    second_layer = activation_function(x, w1)
    print(second_layer)

    w2 = [1, 0.9, 1.5]
    output_layer = neuron.activate(second_layer, w2)
    print(output_layer)
