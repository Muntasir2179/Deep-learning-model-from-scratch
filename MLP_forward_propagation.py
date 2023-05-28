import numpy as np


class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        '''
        -> creating a list where each item represents the number of neurons in that particular layer
        -> for example if we pass num_input=3, num_hidden=[4, 4] and num_output=1
        -> then the list representation will be like [3, 4, 4, 1]
        -> each element representing number of neurons in that layer
        '''
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        # print(layers)

        # initiate random weights
        self.weights = []
        # no weights for the output layer that's why ---> len(layers)-1
        for i in range(len(layers)-1):
            ''' 
            -> creating a 2D matrix
            -> layers[i] is the number of rows or number of neurons in the present layer
            -> layers[i+1] is the number of columns or number of neurons in the subsequent layer
            -> np.random.rand() function generates value in range of (0,1) exclusively
            '''
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

    def forward_propagate(self, inputs):
        # input layer activation/neuron-values
        activations = inputs

        for w in self.weights:
            # calculating net input for nex_layer
            net_inputs = np.dot(activations, w)
            # applying the sigmoid function to calculate activation/neuron-value
            activations = self._sigmoid(net_inputs)
        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # creating model
    model = MLP(
        num_inputs=3,
        num_hidden=[4, 4],
        num_outputs=2
    )

    # creating inputs
    inputs = np.random.rand(model.num_inputs)

    # perform forward propagation
    outputs = model.forward_propagate(inputs)

    # print the result
    print('The network input is: {}'.format(inputs))
    print('The network output is: {}'.format(outputs))
