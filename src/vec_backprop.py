import numpy as np


def vec_forward(x, mlp, t):
    """vectorized forward pass of forward pass with sigmoid activation for hidden layer and softmax for output layer, using cross-entropy loss
        Numpy dot product is used for matrix multiplication of input and weight matrices.
    """
    h = np.dot(x, mlp.W) + mlp.b
    h = 1 / (1 + np.exp(-h))
    y = np.dot(h, mlp.V) + mlp.c
    y_shifted = y - np.max(y)
    exp_y = np.exp(y_shifted)
    y = exp_y / np.sum(exp_y)
    loss = np.log(np.sum(exp_y)) - y_shifted[t]
    return y, h, loss


def vec_forward_batch(X, mlp, t):
    """Forward pass for a batch of inputs."""
    h = 1 / (1 + np.exp(-(X @ mlp.W + mlp.b)))
    logits = h @ mlp.V + mlp.c
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    loss = np.mean(
        -logits_shifted[np.arange(X.shape[0]), t]
        + np.log(np.sum(exp_logits, axis=1))
    )
    return y, h, loss
    
    
def vec_backward(y, h, x, mlp, t):
    """vectorized backward pass
        Numpy outer product is used for computing the outer product of vectors. 
    """
    dL_dy = y.copy()
    dL_dy[t] -= 1  # derivative of cross-entropy loss wrt softmax output
    dL_dV = np.outer(h, dL_dy)  # gradient wrt V
    dL_dc = dL_dy  # gradient wrt c
    
    dL_dh = np.dot(mlp.V, dL_dy) * h * (1 - h)  # derivative of sigmoid activation
    dL_dW = np.outer(x, dL_dh)  # gradient wrt W
    dL_db = dL_dh  # gradient wrt b
    
    return dL_dW, dL_db, dL_dV, dL_dc


def vec_backward_batch(y, h, X, mlp, t):
    """Backward pass for a batch, returning mean gradients."""
    batch_size = X.shape[0]
    dL_dy = y.copy()
    dL_dy[np.arange(batch_size), t] -= 1

    dV = h.T @ dL_dy / batch_size
    dc = np.sum(dL_dy, axis=0) / batch_size
    dL_dh = (dL_dy @ mlp.V.T) * h * (1 - h)
    dW = X.T @ dL_dh / batch_size
    db = np.sum(dL_dh, axis=0) / batch_size

    return dW, db, dV, dc

def vec_update(mlp, dW, db, dV, dc, learning_rate):
    """update weights and biases using gradient descent"""
    mlp.W -= learning_rate * dW
    mlp.b -= learning_rate * db
    mlp.V -= learning_rate * dV
    mlp.c -= learning_rate * dc
