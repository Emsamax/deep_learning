import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# arbitrary points 
X_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
Y_np = np.array([3.0, 5.0, 6.7, 8.4, 13.0], dtype=np.float32) 

# convert points into torch.tensor
X = torch.tensor(X_np)
Y = torch.tensor(Y_np)
N = X.shape[0] 

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


optimizer = optim.SGD([a, b], lr=0.01)

print(f"initials parametres : a={a.item():.4f}, b={b.item():.4f}")

for i in range(100):
    # affine regression model
    Y_ = a * X + b 
    diff = (Y - Y_)
    loss = torch.sum(diff**2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')