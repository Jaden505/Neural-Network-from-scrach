from data_load import load_mnist
from vec_backprop import *

import time
import matplotlib.pyplot as plt
import numpy as np
import random
from dataclasses import dataclass

(xtrain, ytrain), (xval, yval), num_cls = load_mnist()
BATCH_SIZE = 128

@dataclass
class MLP:
    input_size: int
    hidden_size: int
    output_size: int
    
    def __post_init__(self):
        self.W = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.b = np.zeros(self.hidden_size)
        self.V = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.c = np.zeros(self.output_size)
        
def train_mlp(mlp, xtrain, ytrain, learning_rate=0.004, epochs=100):
    history = {'loss': [], 'gradients': []}
    for epoch in range(epochs):
        total_loss = 0.0
        gradients_sums = {'dw': 0, 'db': 0, 'dv': 0, 'dc': 0}
        
        for i, (x, t) in enumerate(zip(xtrain, ytrain)): # show progress bar every 10% of the dataset
            y, h, loss = vec_forward(x, mlp, t)
            dW, db, dV, dc = vec_backward(y, h, x, mlp, t)
            gradients_sums['dw'] += dW
            gradients_sums['db'] += db
            gradients_sums['dv'] += dV
            gradients_sums['dc'] += dc
            
            if i % BATCH_SIZE == 0:
                # mini batch update the gradient sums
                vec_update(mlp, gradients_sums['dw']/BATCH_SIZE, gradients_sums['db']/BATCH_SIZE, 
                           gradients_sums['dv']/BATCH_SIZE, gradients_sums['dc']/BATCH_SIZE, learning_rate)
                gradients_sums = {'dw': 0, 'db': 0, 'dv': 0, 'dc': 0}
            total_loss += loss
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(xtrain)}")
        history['loss'].append(total_loss / len(xtrain))
        history['gradients'].append(gradients_sums)
    return history

def train_mlp_batch(mlp, xtrain, ytrain, learning_rate=0.004, epochs=100):
    history = {'loss': [], 'gradients': []}
    for epoch in range(epochs):
        total_loss = 0.0
        num_samples = len(xtrain)
        
        for start in range(0, num_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_samples)
            X_batch = xtrain[start:end]
            t_batch = ytrain[start:end]

            y, h, loss = vec_forward_batch(X_batch, mlp, t_batch)
            dW, db, dV, dc = vec_backward_batch(y, h, X_batch, mlp, t_batch)
            vec_update(mlp, dW, db, dV, dc, learning_rate)
            total_loss += loss * (end - start)
        
        epoch_loss = total_loss / num_samples
        history['gradients'].append({'dW': dW, 'db': db, 'dV': dV, 'dc': dc})
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")
        history['loss'].append(epoch_loss)
    return history

def plot_loss(history):
    plt.plot(history['loss'])
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
        
if __name__ == "__main__":
    mnist_mlp = MLP(input_size=784, hidden_size=300, output_size=10)    
    history = train_mlp(mnist_mlp, xtrain, ytrain, epochs=1)
    plot_loss(history)
    