import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import time
import torchvision


dataset_root = './mnist_data' 
try:
    print("Loading MNIST")
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)
except Exception:
    print("Download failed")
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=False)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets

# Normalization and Flattening
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)
N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2] 
C = y_train.max().add_(1).item()
x_train_flat = x_train.view(N, D)
x_test_flat = x_test.view(x_test.shape[0], D)
print(f"Data loaded. D={D}, C={C}")

class PTDeep(nn.Module):
    # Using PTDeep
    def __init__(self, config, activation_fn=torch.relu):
        super().__init__()
        self.config = config
        self.activation_fn = activation_fn
        self.layers = nn.ModuleList()

        # Create linear models and initialize weights (as in the previous version)
        for i in range(len(config) - 1):
            Din = config[i]
            Dout = config[i+1]
            linear = nn.Linear(Din, Dout)
            nn.init.xavier_uniform_(linear.weight) 
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)
    
    def count_params(self):
        total_params = 0
        for param in self.named_parameters():
            if param.requires_grad:
                num_params = np.prod(param.size())
                total_params += num_params
        return total_params

    def forward(self, X):
       # Apply layers with activation function
        for i in range(len(self.layers) - 1):
            X = self.layers[i](X)
            X = self.activation_fn(X)
        # Last layer: returns LOGITS (raw output) for compatibility with nn.CrossEntropyLoss
        logits = self.layers[-1](X) 
        return logits

def get_loss(model, X, Y_, param_lambda=1e-3, criterion=nn.CrossEntropyLoss(reduction='mean')):
    """Calculates Cross-Entropy loss + L2 regularization."""
    logits = model(X)
    data_loss = criterion(logits, Y_)
    # Calculate L2 loss only on weights (as required)
    reg_loss = sum(torch.sum(p**2) for name, p in model.named_parameters() if 'weight' in name)
    return data_loss + 0.5 * param_lambda * reg_loss

def train_epoch(model, optimizer, X, Y_, batch_size, param_lambda, criterion):
    """Implement stochastic gradient descent with training on mini-batches."""
    model.train()
    N_data = X.shape[0]
    indices = torch.randperm(N_data)
    total_loss = 0.0
    
    for i in range(0, N_data, batch_size):
        batch_indices = indices[i:i + batch_size]
        X_batch = X[batch_indices]
        Y_batch = Y_[batch_indices]
        
        optimizer.zero_grad()
        loss = get_loss(model, X_batch, Y_batch, param_lambda, criterion)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
        
    return total_loss / N_data

def evaluate(model, X, Y_, param_lambda=0):
    """Evaluates model performance (Accuracy, Loss, Predictions)."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        Y_pred = torch.argmax(logits, dim=1)
        accuracy = (Y_pred == Y_).float().mean().item()
        
        # Calculate loss (with L2 if specified, otherwise without for evaluation)
        loss = get_loss(model, X, Y_, param_lambda=param_lambda).item()
        return accuracy, loss, Y_pred

class KSVMWrap:
    """Wrapper for RBF Kernel SVM."""
    def __init__(self, X, Y_, kernel='rbf'):
        # Using SVC which implements one-vs-one by default for multiclass
        self.model = SVC(kernel=kernel, C=1.0, gamma='scale', probability=False, random_state=42)
        print(f"  Starting SVM training ({kernel})...")
        self.model.fit(X, Y_)
        print("  SVM training finished.")
    def predict(self, X):
        return self.model.predict(X)

def run_experiment(config, X_train, Y_train, X_test, Y_test, epochs, lr, param_lambda, optimizer_type, scheduler_gamma=None, X_val=None, Y_val=None, store_train_metrics=False):
    model = PTDeep(config)
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma) if scheduler_gamma else None
    criterion = nn.CrossEntropyLoss()
    
    loss_history_train = []
    acc_history_test = []
    acc_history_val = []
    
    best_val_acc = -1
    best_model_state = model.state_dict() # Initial state
    best_epoch = 0
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, optimizer, X_train, Y_train, 64, param_lambda, criterion)
        
        if store_train_metrics:
            loss_history_train.append(train_loss)
            
        test_acc, _, Y_pred_test = evaluate(model, X_test, Y_test, param_lambda=0)
        acc_history_test.append(test_acc)

        if scheduler: scheduler.step()
        
        if X_val is not None:
            val_acc, _, _ = evaluate(model, X_val, Y_val, param_lambda=0)
            acc_history_val.append(val_acc)
            
            # Early Stopping (7.D)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                best_epoch = epoch
            
    # Load the best model for final evaluation if Early Stopping was enabled
    if X_val is not None:
        model.load_state_dict(best_model_state)
        print(f"  Final model restored from epoch {best_epoch} (Early Stop).")

    # Final evaluation on train and test sets
    train_acc, train_loss, Y_pred_train = evaluate(model, X_train, Y_train, param_lambda=0)
    test_acc, test_loss, Y_pred_test = evaluate(model, X_test, Y_test, param_lambda=0)

    # Advanced metrics calculation
    Y_tr_np = Y_train.cpu().numpy()
    Y_te_np = Y_test.cpu().numpy()
    Y_pred_tr_np = Y_pred_train.cpu().numpy()
    Y_pred_te_np = Y_pred_test.cpu().numpy()
    
    cm_train = confusion_matrix(Y_tr_np, Y_pred_tr_np)
    cm_test = confusion_matrix(Y_te_np, Y_pred_te_np)
    
    # Precision and Recall: macro average for multiclass
    precision_train = precision_score(Y_tr_np, Y_pred_tr_np, average='macro', zero_division=0)
    recall_train = recall_score(Y_tr_np, Y_pred_tr_np, average='macro', zero_division=0)
    precision_test = precision_score(Y_te_np, Y_pred_te_np, average='macro', zero_division=0)
    recall_test = recall_score(Y_te_np, Y_pred_te_np, average='macro', zero_division=0)
    
    metrics = {
        'train_acc': train_acc, 'train_loss': train_loss, 'train_precision': precision_train, 'train_recall': recall_train, 'cm_train': cm_train,
        'test_acc': test_acc, 'test_loss': test_loss, 'test_precision': precision_test, 'test_recall': recall_test, 'cm_test': cm_test,
        'loss_history_train': loss_history_train, 'acc_history_test': acc_history_test, 'val_acc_history': acc_history_val,
        'time': time.time() - start_time,
        'model': model # Stores the final model for subsequent steps (B continuation)
    }
    
    return metrics


def plot_metrics_comparison(results, metric_key, title):
    """Plots the evolution of loss or accuracy over epochs."""
    plt.figure(figsize=(10, 6))
    for res in results:
        history = res[metric_key]
        if history:
            plt.plot(history, label=res['name'])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric_key.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_weights(model, config, title):
    """Assignment 7.A: Displays the weights of the first layer."""
    if len(config) < 2 or config[1] not in (10, 100): return
    W = model.layers[0].weight.data.cpu().numpy()
    # If the layer is [D, 10], transpose to visualize the 10 classes
    if config[1] == 10: 
        W = W.T 
    
    num_to_plot = min(W.shape[0], 10)
    plt.figure(figsize=(10, 4))
    plt.suptitle(title, fontsize=12)
    for i in range(num_to_plot):
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(W[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        ax.set_title(f"Weight {i}")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_all_assignments(X_tr, Y_tr, X_te, Y_te):
    global D, C 

    N_val = N // 5
    indices = torch.randperm(N)
    X_tr_s, Y_tr_s = X_tr[indices[N_val:]], Y_tr[indices[N_val:]]
    X_val, Y_val = X_tr[indices[:N_val]], Y_tr[indices[:N_val]]
    print(f"Validation set separated: N_train_sub={X_tr_s.shape[0]}, N_val={X_val.shape[0]}")
    
    BEST_CONFIG = [D, 100, 100, C] 
    

    configs_b = [
        [D, C], 
        [D, 100, C], 
        #[D, 100, 100, C], 
        #[D, 100, 100, 100, C] 
    ] 
    results_b = []
    print("\n[B] Architecture Comparison [784,10] to [784,100,10]")
    
    for config in configs_b:
        print(f"  Training {config}...")
        epochs = 20 if len(config) <= 3 else 30 # More epochs for deep models
        lr = 0.1 if len(config) == 2 else 0.05
        
        # Using the full training set for B, but storing losses
        metrics = run_experiment(config, X_tr, Y_tr, X_te, Y_te, epochs, lr, 1e-4, 'SGD', store_train_metrics=True)
        
        results_b.append({
            'name': f"Config {config}",
            'config': config, 
            'lr': lr,
            **metrics
        })
        print(f"  Config {config}: Test Acc={metrics['test_acc']:.4f}, Test Precision={metrics['test_precision']:.4f}, Test Recall={metrics['test_recall']:.4f}")

    # Plot of loss per epoch for model comparison (B)
    plot_metrics_comparison(results_b, 'loss_history_train', "Training Loss per Epoch (Architecture Comparison)")
    
    # Determining the best model 
    best_res = max(results_b, key=lambda x: x['test_acc'])
    print(f"\nBest configuration (B): {best_res['config']} with Test Acc={best_res['test_acc']:.4f}")
    
    # B continuation: Hardest samples for the best model
    model = best_res['model']
    logits = model(X_te); criterion = nn.CrossEntropyLoss(reduction='none')
    losses = criterion(logits, Y_te).detach().cpu().numpy(); hardest_indices = np.argsort(losses)[-9:]
    Y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    plt.figure(figsize=(10, 10)); plt.suptitle(f"9 Hardest Samples for {best_res['config']}");
    for i, idx in enumerate(hardest_indices):
        ax = plt.subplot(3, 3, i + 1); ax.imshow(X_te[idx].cpu().numpy().reshape(28, 28), cmap='gray')
        ax.set_title(f"True: {Y_te[idx].item()}, Pred: {Y_pred[idx]}\nLoss: {losses[idx]:.2f}"); ax.axis('off')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_all_assignments(x_train_flat, y_train, x_test_flat, y_test)