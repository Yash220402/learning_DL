import numpy as np
from random import random


class MLP(object):
    def __init__(self, num_inputs=3, hidden_layers=[3,3], num_outputs=2):
        """
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (int): a list of int for the number of hidden hidden_layers
            num_outputs (int): Number of Outputs
        """
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # a list of int and list to represent the layers
        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]

        # weights matrix for storing weights
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # derivatives
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # activations
        activations = []
        for i in range(len(layers)):
            a = np.zeros((layers[i]))
            activations.append(a)
        self.activations = activations


    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """
        activations = inputs
        self.activations[0] = activations

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)

            activations = self._sigmoid(net_inputs)

            self.activations[i + 1] = activations

        return activations


    def backward_propagate(self, errors):
        """Propagates an error signal backwards.
        Args:
            error (ndarray): the error to backprop
        Return:
            error (ndarray): final error of the inputs 
        """
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            # apply sigmoid derivative functions on each activations
            delta = errors * self._sigmoid_derivative(activations)
            delta_re = delta.reshape(delta.shape[0], -1).T
            # get activations for the current layer
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            # save derivatives after matrix multiplications
            self.derivatives[i] = np.dot(current_activations, delta_re)
            # backpropagate the next error
            error = np.dot(delta, self.weights[i].T)

    
    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model by forward propagation and backward propagation
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): NO. of epochs we want the model to train
            learning_rate (float): step to apply to the gradient descent
        """
        for i in range(epochs):
            sum_errors = 0

            for j, inp in enumerate(inputs):
                target = targets[j]
                output = self.forward_propagate(inp)
                error = target - output

                self.backward_propagate(error)

                # perform gradient descent on the derivatives to update the weights
                self.gradient_descent(learning_rate)

                sum_errors += self._mse(target, output)

            print(f"Epoch: {i+1}\tError: {sum_errors / len(items)}")

        print("\nTraining Complete!")



    def gradient_descent(self, learning_rate=1):
        """Learns by descending the gradient
        """
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate


    # Sigmoid activation function
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


    def _mse(self, target, output):
        return np.average((target - output) ** 2)


if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(items, targets, 50, 0.1)

    # create dummy data
    inputs = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(inputs)

    print(f"Model prediction: {inputs[0]} + {inputs[1]} = {output[0]} \n\t Target = {target[0]}")