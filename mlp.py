import numpy as np

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_deriv(x):
    """Derivative of ReLU."""
    return (x > 0).astype(x.dtype)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)

class MLP:
    def __init__(self, layer_sizes, activations):
        """
        Initializes the MLP.
        
        Parameters:
            layer_sizes: list of integers. For example, [input_dim, hidden1, ..., output_dim].
            activations: list of strings ('relu' or 'sigmoid') for each layer (excluding the input layer).
        """
        assert len(layer_sizes) - 1 == len(activations), "There must be one activation per layer (except input)."
        self.num_layers = len(layer_sizes) - 1
        self.activations = activations
        
        # Initialize weights and biases.
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            # Use He initialization for ReLU or a smaller scale for sigmoid.
            scale = np.sqrt(2 / layer_sizes[i]) if activations[i] == 'relu' else np.sqrt(1 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, X):
        """
        Performs the forward pass.
        
        Parameters:
            X: input data of shape (batch_size, input_dim)
        
        Returns:
            Output of the network.
        """
        self.a_values = [X]  # Store activations (with input as a[0])
        self.z_values = []   # Store pre-activation values
        
        a = X
        for i in range(self.num_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]  # Linear transformation
            self.z_values.append(z)
            # Apply activation
            if self.activations[i] == 'relu':
                a = relu(z)
            elif self.activations[i] == 'sigmoid':
                a = sigmoid(z)
            else:
                raise ValueError("Unsupported activation: " + self.activations[i])
            self.a_values.append(a)
        return a
    
    def backward(self, dL_dy):
        """
        Performs the backward pass.
        
        Parameters:
            dL_dy: gradient of the loss with respect to the network's output, shape (batch_size, output_dim)
            
        Returns:
            grad_weights: list of gradients for each weight matrix.
            grad_biases: list of gradients for each bias.
        """
        grad_weights = [None] * self.num_layers
        grad_biases = [None] * self.num_layers
        
        # Start backpropagation from the output.
        delta = dL_dy  # dL/dy
        
        for i in reversed(range(self.num_layers)):
            z = self.z_values[i]
            # Compute the derivative of the activation.
            if self.activations[i] == 'relu':
                d_activation = relu_deriv(z)
            elif self.activations[i] == 'sigmoid':
                d_activation = sigmoid_deriv(z)
            else:
                raise ValueError("Unsupported activation: " + self.activations[i])
            
            # Elementwise multiply with the activation derivative.
            delta *= d_activation  # Now delta is dL/dz for layer i
            
            # Compute gradients using np.einsum for the outer product between 
            # the previous layer activations and delta.
            a_prev = self.a_values[i]  # shape: (batch_size, n_in)
            grad_w = np.sum(np.einsum('bi,bj->bij', a_prev, delta), axis=0)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            grad_weights[i] = grad_w
            grad_biases[i] = grad_b
            
            # Backpropagate delta to the previous layer.
            delta = np.dot(delta, self.weights[i].T)
            
        return grad_weights, grad_biases
    
    def update_parameters(self, grad_weights, grad_biases, learning_rate):
        """
        Updates weights and biases using gradient descent.
        
        Parameters:
            grad_weights: list of gradients for the weight matrices.
            grad_biases: list of gradients for the bias vectors.
            learning_rate: step size for parameter updates.
        """
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * grad_weights[i]
            self.biases[i] -= learning_rate * grad_biases[i]

# --- Running the 4-layer network defined by the table ---
if __name__ == '__main__':
    # According to the table:
    # Layer 1: Input dimension A, Neurons = 6
    # Layer 2: Input dimension (from layer 1) = 6, Neurons = 4
    # Layer 3: Input dimension (from layer 2) = 4, Neurons = 3
    # Layer 4: Input dimension (from layer 3) = 3, Neurons = 2
    # For demonstration, we choose A = 8.
    input_dim = 8
    layer_sizes = [input_dim, 6, 4, 3, 2]
    
    # Choose activation functions for each layer.
    # For example, use ReLU for hidden layers and Sigmoid for the output.
    activations = ['relu', 'relu', 'relu', 'sigmoid']
    
    # Create the MLP.
    mlp = MLP(layer_sizes, activations)
    
    # Generate a random batch of inputs (e.g., 5 samples).
    X = np.random.randn(5, input_dim)
    print("Input:\n", X)
    
    # Perform a forward pass.
    output = mlp.forward(X)
    print("\nOutput of forward pass:\n", output)
    
    # Assume a dummy gradient from the loss with respect to the network's output.
    dL_dy = np.random.randn(5, 2)
    
    # Compute gradients using backward propagation.
    grad_W, grad_b = mlp.backward(dL_dy)
    print("\nGradient for weights of the first layer:\n", grad_W[0])
    print("\nGradient for biases of the first layer:\n", grad_b[0])
    
    # Update parameters (for example, with a learning rate of 0.01).
    learning_rate = 0.01
    mlp.update_parameters(grad_W, grad_b, learning_rate)