import math


def sigmoid(x):
    """Sigmoid activation function
    Args:
        x (float) : Value to be processed
    Returns:
        y (float) : Output
    """
    
    y = 1.0 / (1 + math.exp(-x))
    return y


def activation(inputs, weights):
    """Computes activation of neurons based on
    input weights.
    Args:
        inputs (list): input signals
        weights (list) : connection weights
    Returns:
        output (float) : output value
    """
    h = 0
    for x, w in zip(inputs, weights):
        # computes the sum of products of input signals and the weights
        h += x*w
        
        
    return sigmoid(h)


if __name__ == "__main__":
    inputs = [0.5, 0.3, 0.2]
    weights = [0.4, 0.7, 0.2]
    output = activation(inputs, weights)
    print(output)
