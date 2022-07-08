# Multilayer perceptron
import numpy as np 

class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        
        # initiate random weights
        self.weights = []
        
        for i in range(len(layers)-1):
            # generating a matrix of random value correspoding to the dimensions
            # w => [3x3] [3x5] [5x2]
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)
            
            
    def forward_prop(self, inputs):
        activations = inputs
            
        for w in self.weights:
            # calculate the net inputs
            net_inputs = np.dot(activations, w)
                
            # calculate the activations
            activations = self._sigmoid(net_inputs)
            
        return activations
    
    
    def _sigmoid(self, x):
        return 1.0 / ( 1 + np.exp(-x))
    
    
if __name__ == "__main__":
    # create an MLP
    mlp = MLP()
    
    # create inputs
    inputs = np.random.rand(mlp.num_inputs)
    
    # perform forward propagation
    outputs = mlp.forward_prop(inputs)
    
    # display results
    print(f"The network input is: {inputs}")
    print(f"The network output is: {outputs}")