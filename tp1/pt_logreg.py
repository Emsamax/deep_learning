import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data import sample_gmm_2d, class_to_onehot, eval_perf_multi, graph_surface, graph_data 

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
            D: dimension of entry data
            C: number of classes
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(D, C) * 0.01)  # w D x C
        self.b = nn.Parameter(torch.zeros(1, C))         # bias 1 x C

    def forward(self, X):
        """
        compute P(Y|X).
        X [NxD]
        Returns: P [NxC]
        """
        Z = torch.mm(X, self.W) + self.b 
        P = torch.softmax(Z, dim=1)
        return P

    def get_loss(self, X, Yoh_, param_lambda=1e-3):
        """
        compute negative log Likelihood + L2 / N
        """
        N = X.shape[0]
        P = self.forward(X)
        log_P = torch.log(P + 1e-12) 
        data_loss = -torch.sum(Yoh_ * log_P) / N 
        #compute loss with L2 reg
        reg_loss = 0.5 * param_lambda * torch.sum(self.W**2)
        return data_loss + reg_loss

def train(model, X, Yoh_, param_iter=1000, param_delta=0.5, param_lambda=1e-3, verbose=True):
    """
    train the model
    """
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    for i in range(param_iter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if param_iter % 50 == 0:
            print(f'Iteration [{i+1}/{param_iter}], Loss: {loss.item():.6f}')

def eval(model, X):
    """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    with torch.no_grad(): 
        P_tensor = model(X_tensor)  
        return P_tensor.detach().numpy()


if __name__ == "__main__":
    np.random.seed(42)
    
    # hyperparam
    N_ITER = 1000
    LR = 0.5
    LAMBDA = 1e-3

    # create data non lineary separable 
    X_np, Y_np = sample_gmm_2d(ncomponents=4, nclasses=2, nsamples=100)
    #one hot encoding
    Yoh_np = class_to_onehot(Y_np) 
    # pytorch tensor
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Yoh_tensor = torch.tensor(Yoh_np, dtype=torch.float32)
   
    X = X_np
    Yoh_ = Yoh_np
    # non one hot lables for eval_perf_multi
    Y_ = Y_np 

    # define the model:
    D = X.shape[1]
    C = Yoh_.shape[1]
    ptlr = PTLogreg(D, C)

    # training
    train(ptlr, X_tensor, Yoh_tensor, N_ITER, LR, param_lambda=LAMBDA)
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)
    accuracy, pr, M = eval_perf_multi(Y, Y_)
    
    print(f"\training accuracy: {accuracy*100:.2f}%")
    print("accuracy by classes:")
    for i in range(C):
        print(f"  Classe {i}: Précision={pr[i][1]:.4f}")

    # visualize the results, decicion surface
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    
    # draw the surface
    def decision_function(X_input):
        probs = eval(ptlr, X_input)
        return probs[:, 0] 

    plt.figure(figsize=(8, 6))
    graph_surface(decision_function, rect, offset=0.5)
    graph_data(X, Y_, Y) 
    plt.title(f" PyTorch logistic reg (Acc={accuracy*100:.2f}%)")
    plt.show()