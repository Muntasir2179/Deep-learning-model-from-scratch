import math


def sigmoid(result):
    return 1.0 / (1 + math.exp(-result))


def activate(inputs, weights):
    result = 0
    for x, w in zip(inputs, weights):
        result = result + (x * w)
    return sigmoid(result)


if __name__ == "__main__":
    inputs = [0.5, 0.3, 0.2]
    weights = [0.4, 0.7, 0.2]
    print(activate(inputs, weights))
