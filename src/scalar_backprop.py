import math

def scalar_forward(x, W, b, V, c, ti):
    # layer 1 linear transformation
    k = [0.0, 0.0, 0.0]
    for j in range(3):
        for i in range(2):
            k[j] += W[i][j] * x[i]
        k[j] += b[j]
    
    # sigmoid activation
    h = [1.0 / (1.0 + math.exp(-k[j])) for j in range(3)]
    
    # layer 2 linear transformation
    o = [0.0, 0.0]
    for m in range(2):
        for j in range(3):
            o[m] += V[j][m] * h[j]
        o[m] += c[m]
        
    # layer 2 softmax activation
    exp_o = [math.exp(o[m]) for m in range(2)]
    y = [exp_o[m] / sum(exp_o) for m in range(2)]
    
    # cross entropy loss
    loss = -1.0 * math.log(y[ti])
    return y, h, loss
    
def scalar_backward(y, h, x, W, V, ti):
    # derivative of cross entropy loss: -1 / y_c
    dy = [0.0, 0.0]
    dy[ti] = -1.0 / y[ti]
    
    # derivative of softmax 
    do = [0.0, 0.0]
    for m in range(2):
        for n in range(2):
            if m == n:
                dy_do = y[n] * (1.0 - y[m])
            else:
                dy_do = -y[n] * y[m]
            do[m] += dy[n] * dy_do
    
    # derivative of layer 2 linear transformation
    dV = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    dc = [0.0, 0.0]
    dh = [0.0, 0.0, 0.0]
    
    for m in range(2):
        dc[m] = do[m]
        for j in range(3):
            dV[j][m] = do[m] * h[j]
            dh[j] += do[m] * V[j][m]
            
    # sigmoid derivative
    dk = [dh[j] * h[j] * (1.0 - h[j]) for j in range(3)]

    # derivative of layer 1 linear transformation
    dW = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    db = [0.0, 0.0, 0.0]

    for j in range(3):
        db[j] = dk[j]
        for i in range(2):
            dW[i][j] = dk[j] * x[i]

    return dW, db, dV, dc

def scalar_update(mlp, dW, db, dV, dc, learning_rate):
    for i in range(mlp.input_size):
        for j in range(mlp.hidden_size):
            mlp.W[i][j] -= learning_rate * dW[i][j]
    for j in range(mlp.hidden_size):
        mlp.b[j] -= learning_rate * db[j]
    for j in range(mlp.hidden_size):
        for k in range(mlp.output_size):
            mlp.V[j][k] -= learning_rate * dV[j][k]
    for k in range(mlp.output_size):
        mlp.c[k] -= learning_rate * dc[k]

if __name__ == "__main__":
    x = [1.0, -1.0]
    W = [[1.0, 1.0, 1.0], 
        [-1.0, 1.0, -1.0]]
    b = [0.0, 0.0, 0.0]

    V = [[1.0, 1.0], 
        [-1.0, -1.0], 
        [-1.0, -1.0]]
    c = [0.0, 0.0]
    ti = 0

    y, h, loss = scalar_forward(x, W, b, V, c, ti)
    dW, db, dV, dc = scalar_backward(y, h, x, W, V, ti)
    
    print("dW, db:", dW, db)
    print("dV, dc:", [[round(val, 4) for val in row] for row in dV], dc)