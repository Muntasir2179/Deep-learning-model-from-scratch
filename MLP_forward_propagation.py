import numpy as np


'''
    -> save activations and derivatives
    -> implement back propagation
    -> implement gradient descent
    -> implement train 
    -> train out network with some dummy dataset
    -> make some predictions
'''


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
        weights = []
        # no weights for the output layer that's why ---> len(layers)-1
        for i in range(len(layers)-1):
            ''' 
            -> creating a 2D matrix
            -> layers[i] is the number of rows or number of neurons in the present layer
            -> layers[i+1] is the number of columns or number of neurons in the subsequent layer
            -> np.random.rand() function generates value in range of (0,1) exclusively
            '''
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # creating a list to store activations
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        # creating a list to store derivatives
        derivatives = []
        for i in range(len(layers)-1):
            # creating a 2D matrix
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        # input layer activation/neuron-values
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculating net input for nex_layer
            net_inputs = np.dot(activations, w)
            # applying the sigmoid function to calculate activation/neuron-value
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
            '''
            -> a_3 = sigmoid(h_3)
            -> h_3 = a_2 * w_2
            '''
        return activations

    def back_propagate(self, error, verbose=False):
        '''
        -> dE/dW_i = (y - a_[i+1])  s'(h_[i+1])  a_i
        -> s'(h_[i+1]) = s(h_[i+1]) (1 - s(h_[i+1]))
        -> s(h_[i+1]) = a_[i+1]

        -> dE/dW_[i-1] = (y - a_[i+1])   s'(h_[i+1])   W_i   s'(h_i)   a_[i-1]
        '''

        # reversed() is used to iterate the layers from right to left
        for i in reversed(range(len(self.derivatives))):
            # i+1 refers to the previous subsequent layer
            activations = self.activations[i+1]

            delta = error * self._sigmoid_derivative(activations)
            # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]
            # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            current_activations_reshaped = current_activations.reshape(
                current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(
                current_activations_reshaped, delta_reshaped)

            # dE/dW_[i-1] = (y - a_[i+1])   s'(h_[i+1])   W_i   s'(h_i)   a_[i-1]
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print('Derivatives for W{}: {}'.format(i, self.derivatives[i]))
        return error

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        # s'(h_[i+1]) = s(h_[i+1]) (1 - s(h_[i+1]))
        # sigmoid() function is previously applied on the x or h_[i+1], that's why we don't need to apply sigmoid here
        return x * (1.0 - x)


if __name__ == '__main__':

    # create an MLP
    mlp = MLP(2, [5], 1)

    # create dummy data
    input = np.array([0.1, 0.2])
    target = np.array([0.3])

    # forward propagation
    output = mlp.forward_propagate(input)

    # calculate error
    error = target - output

    # back propagation
    mlp.back_propagate(error, verbose=True)
