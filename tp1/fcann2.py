# Fichier: fcann2.py
import matplotlib.pyplot as plt
import numpy as np
from data import * 

class FCANN2:
    """ 
    D: dimension of the entry data
    H: size of the hidden layer (1 hidden layer)
    C: nb of classes
    """
    def __init__(self, D, H, C):
        self.D = D
        self.H = H
        self.C = C
        
        # init weight with random value 
        self.W1 = np.random.randn(D, H) * 0.01
        self.b1 = np.zeros((1, H))
        self.W2 = np.random.randn(H, C) * 0.01
        self.b2 = np.zeros((1, C))
        
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def forward(self, X):
        """compute P(Y|X)"""
        # first hidden layer
        self.Z1 = X @ self.W1 + self.b1      
        self.H1 = self.relu(self.Z1)         
        
        # second hidden layer
        self.Z2 = self.H1 @ self.W2 + self.b2
        self.P = self.softmax(self.Z2)       
        return self.P

    def loss(self, X, y, param_lambda=1e-3):
        """compute negative Log-Likelihood (NLL) + L2"""
        N = X.shape[0]
        P = self.forward(X)
        # np.arrange creates array of size N-1 to get all X from the line N
        log_likelihood = -np.log(P[np.arange(N), y])
        data_loss = np.sum(log_likelihood) / N
        
        # derivative L2 reg 
        reg_loss = 0.5 * param_lambda * (np.sum(self.W1*self.W1) + np.sum(self.W2*self.W2))
        return data_loss + reg_loss

    def backward(self, X, y, P, param_lambda):
        """ compute gradient """
        N = X.shape[0]
        # P - Y / N  
        dZ2 = P.copy()
        dZ2[np.arange(N), y] -= 1  
        dZ2 /= N                   
        # dW2/dB2
        dW2 = self.H1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dW2 += param_lambda * self.W2
        # dH1
        dH1 = dZ2 @ self.W2.T
        #dZ1
        dZ1 = dH1 * self.relu_derivative(self.Z1)
        #dw1
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        dW1 += param_lambda * self.W1
        return dW1, db1, dW2, db2

    def train(self, X, y, n_epochs=1000, param_delta=1.0, param_lambda=1e-3):
        """train the model using the gradient decent"""
        for i in range(n_epochs):
            P = self.forward(X)
            dW1, db1, dW2, db2 = self.backward(X, y, P, param_lambda)
            #update the weights and bias
            self.W1 -= param_delta * dW1
            self.b1 -= param_delta * db1
            self.W2 -= param_delta * dW2
            self.b2 -= param_delta * db2
            # print the loss once in 10 000 iteration
            if i%10000 == 0:
                L = self.loss(X, y, param_lambda)
                print(f"Epoch {i}/{n_epochs}, Loss: {L:.6f}")

    def classify(self, X):
        """Predict the class for X"""
        P = self.forward(X)
        return np.argmax(P, axis=1)

if __name__ == '__main__':
    np.random.seed(42) 
    X, Y_ = sample_gmm_2d(ncomponents=6, nclasses=2, nsamples=500)
    D = X.shape[1]  # 2 (because 2D)
    H = 5        # height of the hidden layer
    C = len(np.unique(Y_)) # 2 number of classes
    # declare the model
    model = FCANN2(D, H, C)
    # training the model
    model.train(X, Y_, n_epochs=50000, param_delta=0.001, param_lambda=1e-3,)
    Y = model.classify(X)
    accuracy, pr, M = eval_perf_multi(Y, Y_)
    print(f"\nfinal accuracy on training set: {accuracy*100:.2f}%")
    
    rect = (np.min(X, axis=0), np.max(X, axis=0))

    plt.figure(figsize=(12, 5))
    def decision_fun(X_input):
        P = model.forward(X_input)
        return P[:, 0] 


    graph_surface(decision_fun, rect, offset=0.5)
    # Trace les données avec les classes prédites
    plt.title(f"FCANN2 (Acc: {accuracy*100:.2f}%)")
    plt.show()

