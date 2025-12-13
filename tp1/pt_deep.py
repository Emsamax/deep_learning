import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data import sample_gmm_2d, class_to_onehot, eval_perf_multi, graph_surface, graph_data 


def get_loss(model, X, Yoh_, param_lambda=1e-4):
    N = X.shape[0]
    P = model.forward(X)
    log_P = torch.log(torch.clamp(P, 1e-12, 1.0)) 
    data_loss = -torch.sum(Yoh_ * log_P) / N
    
    reg_loss = 0.0
    for name, param in model.named_parameters():
       # L2 reg only weights not bias
        if 'weight' in name:
            reg_loss += torch.sum(param**2)
    total_reg_loss = 0.5 * param_lambda * reg_loss
    return data_loss + total_reg_loss

def train(model, X, Yoh_, param_niter=1000, param_delta=0.1, param_lambda=1e-4, verbose=True):
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    for i in range(param_niter):
        model.train()
        loss = get_loss(model, X, Yoh_, param_lambda)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f'  Iteration [{i+1}/{param_niter}], Loss: {loss.item():.6f}')

def eval(model, X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval() 
    with torch.no_grad(): 
        P_tensor = model(X_tensor) 
        return P_tensor.detach().numpy()

class PTDeep(nn.Module):
    def __init__(self, config, activation_fn=torch.relu):
        super().__init__()
        self.config = config
        self.activation_fn = activation_fn
        self.layers = nn.ModuleList()

        # create linear models
        for i in range(len(config) - 1):
            Din = config[i]
            Dout = config[i+1]
            self.layers.append(nn.Linear(Din, Dout))
    
    def count_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = np.prod(param.size())
                print(f"  {name}: {list(param.size())} -> {num_params} params")
                total_params += num_params
        print(f"number of params for training: {total_params}")
        return total_params

    def forward(self, X):
       # -1 to not itearte on last layer 
        for i in range(len(self.layers) - 1):
            X = self.layers[i](X)
            X = self.activation_fn(X)
        # last layer for different activation function 
        logits = self.layers[-1](X) 
        P = torch.softmax(logits, dim=1)
        return P

class PTDeepBN(nn.Module):
    def __init__(self, config, activation_fn=torch.relu):
        super().__init__()
        self.config = config
        self.activation_fn = activation_fn
        
        self.linear_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList() 

        # -1 to not iterate on the last layer
        for i in range(len(config) - 1):
            Din = config[i]
            Dout = config[i+1]
            
            self.linear_layers.append(nn.Linear(Din, Dout))
            if i < len(config) - 2:
                self.batch_norm_layers.append(nn.BatchNorm1d(Dout))

    def count_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = np.prod(param.size())
                print(f"  {name}: {list(param.size())} -> {num_params} params")
                total_params += num_params
        print(f"number of params for training:{total_params}")
        return total_params

    def forward(self, X):
        for i in range(len(self.batch_norm_layers)):
            X = self.linear_layers[i](X)
            X = self.batch_norm_layers[i](X)
            X = self.activation_fn(X)
            
        #last linear layer is at len(config) -2
        logits = self.linear_layers[-1](X) 
        P = torch.softmax(logits, dim=1)
        return P


if __name__ == "__main__":
    np.random.seed(42) 
    N_SAMPLES = 10    
    N_COMPONENTS = 6   
    N_CLASSES = 2        
    """
    N_SAMPLES = 40  
    N_COMPONENTS = 4
    N_CLASSES = 2   
    """      
    
    # profound model
    CONFIG = [2, 10, 10, 2]
    ACTIVATION_FN = torch.relu
    
    N_ITER = 10000    
    LR = 0.1          # learning rate
    LAMBDA = 1e-4     # tiny L2 reg factor 
    
    X_np, Y_np = sample_gmm_2d(ncomponents=N_COMPONENTS, nclasses=N_CLASSES, nsamples=N_SAMPLES)
    Yoh_np = class_to_onehot(Y_np) 
    #convert to tensor pytorch
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Yoh_tensor = torch.tensor(Yoh_np, dtype=torch.float32)
    
    print("\n=======================================================")
    print(f"1. training without bn : config - {CONFIG}")
    ptdeep = PTDeep(CONFIG, activation_fn=ACTIVATION_FN)
    ptdeep.count_params()
    train(ptdeep, X_tensor, Yoh_tensor, N_ITER, LR, LAMBDA, verbose=True)
    probs_deep = eval(ptdeep, X_np)
    Y_deep = np.argmax(probs_deep, axis=1)
    acc_deep, pr_deep, _ = eval_perf_multi(Y_deep, Y_np)
    print(f"Accuracy: {acc_deep*100:.2f}%")

    print("\n=======================================================")
    print(f"2. training with bn : config - {CONFIG}")
    ptdeep_bn = PTDeepBN(CONFIG, activation_fn=ACTIVATION_FN)
    ptdeep_bn.count_params()
    train(ptdeep_bn, X_tensor, Yoh_tensor, N_ITER, LR, LAMBDA, verbose=True)
    probs_bn = eval(ptdeep_bn, X_np)
    Y_bn = np.argmax(probs_bn, axis=1)
    acc_bn, pr_bn, _ = eval_perf_multi(Y_bn, Y_np)
    print(f"Accuracy : {acc_bn*100:.2f}%")

    #plot results into 2 separate rectangles to see
    rect = (np.min(X_np, axis=0), np.max(X_np, axis=0))
    plt.figure(figsize=(16, 6))

    # subplot 1 without bn
    plt.subplot(1, 2, 1)
    def decision_deep(X_input): return eval(ptdeep, X_input)[:, 0]
    graph_surface(decision_deep, rect, offset=0.5)
    graph_data(X_np, Y_np, Y_deep) 
    plt.title(f"Without BN (Acc: {acc_deep*100:.2f}%, ReLU)")

    # subplot 1 with bn
    plt.subplot(1, 2, 2)
    def decision_bn(X_input): return eval(ptdeep_bn, X_input)[:, 0]
    graph_surface(decision_bn, rect, offset=0.5)
    graph_data(X_np, Y_np, Y_bn) 
    plt.title(f"With BN (Acc: {acc_bn*100:.2f}%, ReLU)")
    plt.show()