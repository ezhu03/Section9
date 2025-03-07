import numpy as np
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================
# MNIST Data Loader (provided)
# ============================
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
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
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images.append(img)
            
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)        

# -----------------------------
# File paths for MNIST (adjust as needed)
# -----------------------------
input_path = 'archive'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# -----------------------------
# Load MNIST and combine training and test sets for an 80-20 split
# -----------------------------
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                   test_images_filepath, test_labels_filepath)
(x_train_list, y_train_list), (x_test_list, y_test_list) = mnist_dataloader.load_data()

# Combine all images and labels
X_total = np.array(x_train_list + x_test_list, dtype=np.float32)  # shape (num_samples, 28,28)
y_total = np.array(list(y_train_list) + list(y_test_list), dtype=np.int32)

# Normalize images to [0, 1]
X_total /= 255.0

# Shuffle and split 80-20
indices = np.arange(X_total.shape[0])
np.random.shuffle(indices)
split_idx = int(0.8 * len(indices))
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X_total[train_indices]
X_test = X_total[test_indices]
y_train = y_total[train_indices]
y_test = y_total[test_indices]

# -----------------------------
# Helper: one-hot encoding
# -----------------------------
def one_hot(y, num_classes=10):
    one_hot_labels = np.zeros((len(y), num_classes))
    one_hot_labels[np.arange(len(y)), y] = 1
    return one_hot_labels

y_train_oh = one_hot(y_train)
y_test_oh  = one_hot(y_test)

# -----------------------------
# Loss and Softmax functions
# -----------------------------
def softmax(x):
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / np.sum(ex, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true):
    epsilon = 1e-12
    return -np.mean(np.sum(y_true * np.log(probs + epsilon), axis=1))

# -----------------------------
# Activation functions and their derivatives
# -----------------------------
def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    return (x > 0).astype(x.dtype)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# -----------------------------
# MLP Class (modified to support "linear" activation)
# -----------------------------
class MLP:
    def __init__(self, layer_sizes, activations):
        assert len(layer_sizes)-1 == len(activations), "Activation for each layer required."
        self.num_layers = len(layer_sizes)-1
        self.activations = activations
        
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            if activations[i] == 'relu' or activations[i] == 'linear':
                scale = np.sqrt(2 / layer_sizes[i])
            else:
                scale = np.sqrt(1 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, X):
        self.a_values = [X]
        self.z_values = []
        a = X
        for i in range(self.num_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if self.activations[i] == 'relu':
                a = relu(z)
            elif self.activations[i] == 'sigmoid':
                a = sigmoid(z)
            elif self.activations[i] == 'linear':
                a = z
            else:
                raise ValueError("Unsupported activation")
            self.a_values.append(a)
        return a
    
    def backward(self, dL_dy):
        grad_weights = [None] * self.num_layers
        grad_biases = [None] * self.num_layers
        
        delta = dL_dy
        for i in reversed(range(self.num_layers)):
            z = self.z_values[i]
            if self.activations[i] == 'relu':
                d_act = relu_deriv(z)
            elif self.activations[i] == 'sigmoid':
                d_act = sigmoid_deriv(z)
            elif self.activations[i] == 'linear':
                d_act = 1.0
            else:
                raise ValueError("Unsupported activation")
            
            delta = delta * d_act
            a_prev = self.a_values[i]
            grad_w = np.sum(np.einsum('bi,bj->bij', a_prev, delta), axis=0)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            grad_weights[i] = grad_w
            grad_biases[i] = grad_b
            
            delta = np.dot(delta, self.weights[i].T)
        return grad_weights, grad_biases
    
    def update_parameters(self, grad_weights, grad_biases, learning_rate):
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * grad_weights[i]
            self.biases[i] -= learning_rate * grad_biases[i]

# -----------------------------
# Vectorized Convolution: im2col and col2im helpers
# -----------------------------
def conv_forward(X, W, b):
    """
    Vectorized convolution forward using im2col.
    """
    N, C, H, W_in = X.shape
    F, _, KH, KW = W.shape
    H_out = H - KH + 1
    W_out = W_in - KW + 1

    # Create a strided view to extract patches
    shape = (N, C, H_out, W_out, KH, KW)
    strides = X.strides[:2] + (X.strides[2], X.strides[3], X.strides[2], X.strides[3])
    X_strided = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    X_col = X_strided.reshape(N, C, H_out, W_out, KH * KW)
    X_col = X_col.transpose(0, 2, 3, 1, 4).reshape(N * H_out * W_out, C * KH * KW)
    
    W_col = W.reshape(F, -1).T  # shape (C*KH*KW, F)
    out_col = X_col.dot(W_col) + b  # broadcasting b (shape (F,))
    out = out_col.reshape(N, H_out, W_out, F).transpose(0, 3, 1, 2)
    cache = (X, W, b, X_col, H_out, W_out, KH, KW)
    return out, cache

def col2im(dX_col, X_shape, KH, KW):
    """
    Convert column gradients back to image gradients.
    """
    N, C, H, W = X_shape
    H_out = H - KH + 1
    W_out = W - KW + 1
    dX = np.zeros(X_shape, dtype=dX_col.dtype)
    dX_col_reshaped = dX_col.reshape(N, H_out, W_out, C, KH, KW).transpose(0, 3, 1, 2, 4, 5)
    for i in range(KH):
        for j in range(KW):
            dX[:, :, i:i+H_out, j:j+W_out] += dX_col_reshaped[:, :, :, :, i, j]
    return dX

def conv_backward(dout, cache):
    """
    Vectorized convolution backward using cached im2col.
    """
    X, W, b, X_col, H_out, W_out, KH, KW = cache
    N, C, H, W_in = X.shape
    F, _, _, _ = W.shape

    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, F)
    
    dW_col = X_col.T.dot(dout_reshaped)  # (C*KH*KW, F)
    dW = dW_col.T.reshape(W.shape)
    
    db = np.sum(dout_reshaped, axis=0)
    
    W_col = W.reshape(F, -1)
    dX_col = dout_reshaped.dot(W_col)
    
    dX = col2im(dX_col, X.shape, KH, KW)
    return dX, dW, db

# -----------------------------
# CNN Class (using vectorized convolution)
# -----------------------------
class CNN:
    def __init__(self, conv_configs, fc_config, conv_activation='relu', fc_activation='linear'):
        self.conv_activation = conv_activation
        self.fc_activation = fc_activation
        
        self.num_conv = len(conv_configs)
        self.conv_weights = []
        self.conv_biases = []
        self.conv_caches = []
        self.conv_pre_acts = []
        
        for cfg in conv_configs:
            num_filters = cfg['num_filters']
            filter_size = cfg['filter_size']
            input_channels = cfg['input_channels']
            # Use He initialization for ReLU activation
            if self.conv_activation == 'relu':
                weight_scale = np.sqrt(2.0 / (input_channels * filter_size * filter_size))
            else:
                weight_scale = 0.01
            W = np.random.randn(num_filters, input_channels, filter_size, filter_size) * weight_scale
            b = np.zeros(num_filters)
            self.conv_weights.append(W)
            self.conv_biases.append(b)
        
        self.fc_input_dim = fc_config['input_dim']
        self.fc_output_dim = fc_config['output_dim']
        self.fc_W = np.random.randn(self.fc_input_dim, self.fc_output_dim) * np.sqrt(2/self.fc_input_dim)
        self.fc_b = np.zeros((1, self.fc_output_dim))
        self.fc_cache = None

    def conv_forward_layer(self, X, W, b):
        out, cache = conv_forward(X, W, b)
        pre_act = out.copy()
        if self.conv_activation == 'relu':
            A = relu(out)
        elif self.conv_activation == 'sigmoid':
            A = sigmoid(out)
        else:
            raise ValueError("Unsupported conv activation")
        return A, cache, pre_act

    def fc_forward(self, X):
        Z = np.dot(X, self.fc_W) + self.fc_b
        A = Z  # linear activation; softmax is applied externally.
        self.fc_cache = (X, Z)
        return A

    def forward(self, X):
        self.conv_caches = []
        self.conv_pre_acts = []
        A = X
        for i in range(self.num_conv):
            A, cache, pre_act = self.conv_forward_layer(A, self.conv_weights[i], self.conv_biases[i])
            self.conv_caches.append(cache)
            self.conv_pre_acts.append(pre_act)
        self.conv_output_shape = A.shape
        N = A.shape[0]
        A_flat = A.reshape(N, -1)
        out = self.fc_forward(A_flat)
        return out

    def fc_backward(self, d_out):
        X, Z = self.fc_cache
        dZ = d_out  # derivative is 1 for linear activation
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.fc_W.T)
        self.fc_grad = (dW, db)
        return dX

    def conv_backward_layer(self, dA, cache, pre_act):
        if self.conv_activation == 'relu':
            dZ = dA * relu_deriv(pre_act)
        elif self.conv_activation == 'sigmoid':
            dZ = dA * sigmoid_deriv(pre_act)
        else:
            dZ = dA
        dX, dW, db = conv_backward(dZ, cache)
        return dX, dW, db

    def backward(self, d_loss):
        dA_flat = self.fc_backward(d_loss)
        dA = dA_flat.reshape(self.conv_output_shape)
        self.conv_grads = []
        for i in reversed(range(self.num_conv)):
            cache = self.conv_caches[i]
            pre_act = self.conv_pre_acts[i]
            dA, dW, db = self.conv_backward_layer(dA, cache, pre_act)
            self.conv_grads.insert(0, (dW, db))
        return

    def update_parameters(self, learning_rate):
        dW_fc, db_fc = self.fc_grad
        self.fc_W -= learning_rate * dW_fc
        self.fc_b -= learning_rate * db_fc
        for i in range(self.num_conv):
            dW, db = self.conv_grads[i]
            self.conv_weights[i] -= learning_rate * dW
            self.conv_biases[i] -= learning_rate * db

# -----------------------------
# Training utilities
# -----------------------------
def iterate_minibatches(X, y, batch_size=64):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

# -----------------------------
# Training loop for MLP
# -----------------------------
def train_mlp(mlp, X_train, y_train_oh, X_test, y_test_oh, num_epochs=20, batch_size=64, learning_rate=0.01):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        total = 0
        for X_batch, y_batch in iterate_minibatches(X_train_flat, y_train_oh, batch_size):
            logits = mlp.forward(X_batch)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss * X_batch.shape[0]
            predictions = np.argmax(probs, axis=1)
            labels = np.argmax(y_batch, axis=1)
            epoch_correct += np.sum(predictions == labels)
            total += X_batch.shape[0]
            
            grad = (probs - y_batch) / X_batch.shape[0]
            grad_W, grad_b = mlp.backward(grad)
            mlp.update_parameters(grad_W, grad_b, learning_rate)
        
        train_losses.append(epoch_loss/total)
        train_acc.append(epoch_correct/total)
        
        test_loss = 0
        correct = 0
        total_test = X_test_flat.shape[0]
        for X_batch, y_batch in iterate_minibatches(X_test_flat, y_test_oh, batch_size):
            logits = mlp.forward(X_batch)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y_batch)
            test_loss += loss * X_batch.shape[0]
            preds = np.argmax(probs, axis=1)
            labs = np.argmax(y_batch, axis=1)
            correct += np.sum(preds == labs)
        test_losses.append(test_loss/total_test)
        test_acc.append(correct/total_test)
        
        print(f"MLP Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Train Acc={train_acc[-1]:.4f} | Test Loss={test_losses[-1]:.4f}, Test Acc={test_acc[-1]:.4f}")
    
    return train_losses, test_losses, train_acc, test_acc

# -----------------------------
# Training loop for CNN
# -----------------------------
def train_cnn(cnn, X_train, y_train_oh, X_test, y_test_oh, num_epochs=20, batch_size=64, learning_rate=0.01):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    
    X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test_cnn  = X_test.reshape(X_test.shape[0], 1, 28, 28)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        total = 0
        for X_batch, y_batch in iterate_minibatches(X_train_cnn, y_train_oh, batch_size):
            logits = cnn.forward(X_batch)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss * X_batch.shape[0]
            preds = np.argmax(probs, axis=1)
            labs = np.argmax(y_batch, axis=1)
            epoch_correct += np.sum(preds == labs)
            total += X_batch.shape[0]
            
            grad = (probs - y_batch) / X_batch.shape[0]
            cnn.backward(grad)
            cnn.update_parameters(learning_rate)
            
        train_losses.append(epoch_loss/total)
        train_acc.append(epoch_correct/total)
        
        test_loss = 0
        correct = 0
        total_test = X_test_cnn.shape[0]
        for X_batch, y_batch in iterate_minibatches(X_test_cnn, y_test_oh, batch_size):
            logits = cnn.forward(X_batch)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y_batch)
            test_loss += loss * X_batch.shape[0]
            preds = np.argmax(probs, axis=1)
            labs = np.argmax(y_batch, axis=1)
            correct += np.sum(preds == labs)
        test_losses.append(test_loss/total_test)
        test_acc.append(correct/total_test)
        
        print(f"CNN Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Train Acc={train_acc[-1]:.4f} | Test Loss={test_losses[-1]:.4f}, Test Acc={test_acc[-1]:.4f}")
    
    return train_losses, test_losses, train_acc, test_acc

# -----------------------------
# Instantiate and train the MLP
# -----------------------------
mlp = MLP(layer_sizes=[784, 128, 64, 10], activations=['relu', 'relu', 'linear'])
print("Training MLP on MNIST...")
mlp_train_losses, mlp_test_losses, mlp_train_acc, mlp_test_acc = train_mlp(mlp, X_train, y_train_oh, X_test, y_test_oh,
                                                                           num_epochs=10, batch_size=128, learning_rate=0.01)

mlp_size = 10
epochs = np.arange(1, mlp_size+1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, mlp_train_losses, label="MLP Train Loss")
plt.plot(epochs, mlp_test_losses, label="MLP Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Loss vs Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, mlp_train_acc, label="MLP Train Accuracy")
plt.plot(epochs, mlp_test_acc, label="MLP Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy vs Epoch")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('MLP_convergence.png')
# -----------------------------
# Compute and plot confusion matrix for test set (using the better network, e.g., CNN)
# -----------------------------
# 1) Flatten the test images for the MLP
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 2) Forward pass: get logits
logits = mlp.forward(X_test_flat)

# 3) Convert logits to predicted labels
probs = softmax(logits)
predictions = np.argmax(probs, axis=1)

# 4) Compute confusion matrix
cm_mlp = confusion_matrix(y_test, predictions)

# 5) Plot confusion matrix with seaborn
plt.figure(figsize=(8,6))
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("MLP Confusion Matrix on MNIST Test Set")
plt.show()
plt.savefig('MLP_confusion_matrix.png')

# -----------------------------
# Instantiate and train the CNN
# -----------------------------
conv_configs = [
    {'num_filters': 8, 'filter_size': 3, 'input_channels': 1},
    {'num_filters': 16, 'filter_size': 3, 'input_channels': 8},
    {'num_filters': 32, 'filter_size': 3, 'input_channels': 16}
]
fc_input_dim = 32 * 22 * 22
fc_config = {'input_dim': fc_input_dim, 'output_dim': 10}

cnn = CNN(conv_configs, fc_config, conv_activation='relu', fc_activation='linear')
print("\nTraining CNN on MNIST...")
cnn_train_losses, cnn_test_losses, cnn_train_acc, cnn_test_acc = train_cnn(cnn, X_train, y_train_oh, X_test, y_test_oh,
                                                                           num_epochs=2, batch_size=128, learning_rate=0.01)

# -----------------------------
# Plot convergence curves for MLP and CNN
# -----------------------------
cnn_size=2
epochs = np.arange(1, cnn_size+1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, cnn_train_losses, label="CNN Train Loss")
plt.plot(epochs, cnn_test_losses, label="CNN Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss vs Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, cnn_train_acc, label="CNN Train Accuracy")
plt.plot(epochs, cnn_test_acc, label="CNN Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy vs Epoch")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('CNN_convergence.png')



# 1) Reshape test images for the CNN
X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28)

# 2) Forward pass: get logits
logits = cnn.forward(X_test_cnn)

# 3) Convert logits to predicted labels
probs = softmax(logits)
predictions = np.argmax(probs, axis=1)

# 4) Compute confusion matrix
cm_cnn = confusion_matrix(y_test, predictions)

# 5) Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("CNN Confusion Matrix on MNIST Test Set")
plt.show()
plt.savefig('CNN_confusion_matrix.png')
