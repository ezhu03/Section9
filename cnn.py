#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        
    
import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = 'archive'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#
# Show some random training and test images 
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)
plt.savefig('MNIST-samples.png')

import numpy as np

# ---------------------------
# Activation Functions
# ---------------------------
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(x.dtype)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# ---------------------------
# Convolution Forward & Backward
# ---------------------------
def conv_forward(X, W, b):
    """
    Naive convolution forward pass.
    X: Input data, shape (N, C, H, W)
    W: Filters, shape (F, C, KH, KW)
    b: Biases, shape (F,)
    
    Returns:
      out: Output data, shape (N, F, H_out, W_out)
      cache: Tuple (X, W, b) for backward pass
    """
    N, C, H, W_in = X.shape
    F, _, KH, KW = W.shape
    H_out = H - KH + 1
    W_out = W_in - KW + 1
    out = np.zeros((N, F, H_out, W_out))
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, f, i, j] = np.sum(X[n, :, i:i+KH, j:j+KW] * W[f]) + b[f]
    cache = (X, W, b)
    return out, cache

def conv_backward(dout, cache):
    """
    Naive convolution backward pass.
    dout: Upstream gradients, shape (N, F, H_out, W_out)
    cache: Tuple (X, W, b) from conv_forward
    
    Returns:
      dX: Gradient w.r.t. input X, shape (N, C, H, W)
      dW: Gradient w.r.t. filters W, shape (F, C, KH, KW)
      db: Gradient w.r.t. biases b, shape (F,)
    """
    X, W, b = cache
    N, C, H, W_in = X.shape
    F, _, KH, KW = W.shape
    H_out = H - KH + 1
    W_out = W_in - KW + 1
    
    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    dW[f] += dout[n, f, i, j] * X[n, :, i:i+KH, j:j+KW]
                    dX[n, :, i:i+KH, j:j+KW] += dout[n, f, i, j] * W[f]
                    db[f] += dout[n, f, i, j]
    return dX, dW, db

# ---------------------------
# CNN Class Definition
# ---------------------------
class CNN:
    def __init__(self, conv_configs, fc_config, conv_activation='relu', fc_activation='sigmoid'):
        """
        Initializes a three-layer CNN with a final fully connected (FC) layer.
        
        Parameters:
          conv_configs: List of dicts for each conv layer, each dict should contain:
              - 'num_filters': number of filters in the layer
              - 'filter_size': size of the (square) filter (K)
              - 'input_channels': number of channels for input to this layer
          fc_config: Dict for the fully connected layer, containing:
              - 'input_dim': dimension after flattening the last conv layer output
              - 'output_dim': number of output neurons (e.g., classes)
          conv_activation: Activation for conv layers ('relu' or 'sigmoid')
          fc_activation: Activation for the FC layer ('relu' or 'sigmoid')
        """
        self.conv_activation = conv_activation
        self.fc_activation = fc_activation
        
        self.num_conv = len(conv_configs)
        self.conv_weights = []
        self.conv_biases = []
        self.conv_caches = []  # to store forward pass caches for conv layers
        self.conv_pre_activations = []  # to store pre-activation outputs
        
        # Initialize convolutional layers
        for cfg in conv_configs:
            num_filters = cfg['num_filters']
            filter_size = cfg['filter_size']
            input_channels = cfg['input_channels']
            W = np.random.randn(num_filters, input_channels, filter_size, filter_size) * 0.01
            b = np.zeros(num_filters)
            self.conv_weights.append(W)
            self.conv_biases.append(b)
        
        # Initialize fully connected layer parameters
        self.fc_input_dim = fc_config['input_dim']
        self.fc_output_dim = fc_config['output_dim']
        self.fc_W = np.random.randn(self.fc_input_dim, self.fc_output_dim) * np.sqrt(2/self.fc_input_dim)
        self.fc_b = np.zeros((1, self.fc_output_dim))
        self.fc_cache = None

    def conv_forward_layer(self, X, W, b):
        """
        Forward pass for one conv layer followed by activation.
        """
        out, cache = conv_forward(X, W, b)
        # Save pre-activation output for backward
        pre_act = out.copy()
        if self.conv_activation == 'relu':
            A = relu(out)
        elif self.conv_activation == 'sigmoid':
            A = sigmoid(out)
        else:
            raise ValueError("Unsupported conv activation")
        return A, cache, pre_act

    def fc_forward(self, X):
        """
        Fully connected forward pass.
        X: input of shape (N, D)
        """
        Z = np.dot(X, self.fc_W) + self.fc_b
        if self.fc_activation == 'relu':
            A = relu(Z)
        elif self.fc_activation == 'sigmoid':
            A = sigmoid(Z)
        else:
            A = Z  # linear
        self.fc_cache = (X, Z)
        return A

    def forward(self, X):
        """
        Forward pass for the CNN.
        X: input data, shape (N, C, H, W)
        Returns:
          out: final output from the FC layer.
        """
        self.conv_caches = []
        self.conv_pre_activations = []
        A = X
        # Forward pass through conv layers
        for i in range(self.num_conv):
            A, cache, pre_act = self.conv_forward_layer(A, self.conv_weights[i], self.conv_biases[i])
            self.conv_caches.append(cache)
            self.conv_pre_activations.append(pre_act)
        # Save conv output shape for backward
        self.conv_output_shape = A.shape  # (N, F, H_out, W_out)
        # Flatten the conv output
        N = A.shape[0]
        A_flat = A.reshape(N, -1)
        # Forward pass through fully connected layer
        out = self.fc_forward(A_flat)
        return out

    def fc_backward(self, d_out):
        """
        Backward pass for the FC layer.
        d_out: gradient of loss w.r.t. FC output, shape (N, fc_output_dim)
        Returns:
          dX: gradient w.r.t. FC input, shape (N, fc_input_dim)
        """
        X, Z = self.fc_cache
        if self.fc_activation == 'relu':
            dZ = d_out * relu_deriv(Z)
        elif self.fc_activation == 'sigmoid':
            dZ = d_out * sigmoid_deriv(Z)
        else:
            dZ = d_out
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.fc_W.T)
        self.fc_grad = (dW, db)
        return dX

    def conv_backward_layer(self, dA, cache, pre_act):
        """
        Backward pass for one conv layer.
        dA: gradient w.r.t. activation output of the conv layer.
        cache: cache from conv_forward
        pre_act: pre-activation output (before applying activation)
        Returns:
          dX, dW, db for this conv layer.
        """
        if self.conv_activation == 'relu':
            dZ = dA * relu_deriv(pre_act)
        elif self.conv_activation == 'sigmoid':
            dZ = dA * sigmoid_deriv(pre_act)
        else:
            dZ = dA
        dX, dW, db = conv_backward(dZ, cache)
        return dX, dW, db

    def backward(self, d_loss):
        """
        Backward pass for the entire CNN.
        d_loss: gradient of loss w.r.t. final output.
        """
        # Backprop through FC layer
        dA_flat = self.fc_backward(d_loss)  # shape (N, fc_input_dim)
        # Reshape to conv output shape
        dA = dA_flat.reshape(self.conv_output_shape)
        # Backprop through conv layers (reverse order)
        self.conv_grads = []
        for i in reversed(range(self.num_conv)):
            cache = self.conv_caches[i]
            pre_act = self.conv_pre_activations[i]
            dA, dW, db = self.conv_backward_layer(dA, cache, pre_act)
            self.conv_grads.insert(0, (dW, db))  # store gradients for layer i
        return

    def update_parameters(self, learning_rate):
        """
        Update parameters for both the FC and conv layers using gradient descent.
        """
        # Update FC parameters
        dW_fc, db_fc = self.fc_grad
        self.fc_W -= learning_rate * dW_fc
        self.fc_b -= learning_rate * db_fc
        
        # Update conv layer parameters
        for i in range(self.num_conv):
            dW, db = self.conv_grads[i]
            self.conv_weights[i] -= learning_rate * dW
            self.conv_biases[i] -= learning_rate * db

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == '__main__':
    # Define configuration for three conv layers.
    # For example, assume an input with 3 channels (RGB image).
    conv_configs = [
        {'num_filters': 8, 'filter_size': 3, 'input_channels': 3},  # Conv Layer 1
        {'num_filters': 16, 'filter_size': 3, 'input_channels': 8}, # Conv Layer 2
        {'num_filters': 32, 'filter_size': 3, 'input_channels': 16} # Conv Layer 3
    ]
    # Assume that after three conv layers (with no padding and stride=1),
    # the output shape becomes (N, 32, H_out, W_out). For this example, we set:
    H_out, W_out = 6, 6  # example output height and width after conv layers
    fc_input_dim = 32 * H_out * W_out
    fc_config = {'input_dim': fc_input_dim, 'output_dim': 10}  # e.g., 10 classes

    # Create CNN instance (using ReLU for conv layers and Sigmoid for FC layer, as an example)
    cnn = CNN(conv_configs, fc_config, conv_activation='relu', fc_activation='sigmoid')
    
    # Generate a random batch of 2 images of size 8x8 with 3 channels.
    X = np.random.randn(2, 3, 8, 8)
    # Forward pass
    out = cnn.forward(X)
    print("CNN output:\n", out)
    
    # Assume a dummy gradient from the loss function
    d_loss = np.random.randn(*out.shape)
    # Backward pass
    cnn.backward(d_loss)
    # Update parameters
    cnn.update_parameters(learning_rate=0.01)
    print("Parameters updated.")
