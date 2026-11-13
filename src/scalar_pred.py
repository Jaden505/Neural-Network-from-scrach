from data_load import load_synth
from scalar_backprop import scalar_forward, scalar_backward, scalar_update

import matplotlib.pyplot as plt
import random
from dataclasses import dataclass

(xtrain, ytrain), (xval, yval), num_cls = load_synth()


@dataclass
class MLP:
    input_size: int
    hidden_size: int
    output_size: int

    def __post_init__(self):
        self.W = [[random.uniform(-1, 1) for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        self.b = [0.0 for _ in range(self.hidden_size)]
        self.V = [[random.uniform(-1, 1) for _ in range(self.output_size)] for _ in range(self.hidden_size)]
        self.c = [0.0 for _ in range(self.output_size)]
        
def train_mlp(mlp, xtrain, ytrain, learning_rate=0.0003, epochs=100):
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
    
        for i, (x, t) in enumerate(zip(xtrain, ytrain)): # show progress bar every 10% of the dataset
            if mlp.model_type == "scalar":
                y, h, loss = scalar_forward(x, mlp.W, mlp.b, mlp.V, mlp.c, t)
                dW, db, dV, dc = scalar_backward(y, h, x, mlp.W, mlp.V, t)
                scalar_update(mlp, dW, db, dV, dc, learning_rate)

            total_loss += loss
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(xtrain)}")
        history.append(total_loss / len(xtrain))
    return history
        
def plot_loss(history):
    plt.plot(history)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
        
if __name__ == "__main__":
    synth_mlp = MLP(input_size=2, hidden_size=3, output_size=2)
    history = train_mlp(synth_mlp, xtrain, ytrain)
    plot_loss(history)
    