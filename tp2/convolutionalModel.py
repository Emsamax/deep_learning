import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.datasets import MNIST
import skimage as ski
import skimage.io

 
IN_CHANNELS = 1
CONV1_WIDTH = 16
CONV2_WIDTH = 32
FC1_WIDTH = 512
CLASS_COUNT = 10
KERNEL_SIZE = 5

class ConvolutionalModel(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, conv1_width=CONV1_WIDTH, 
                 conv2_width=CONV2_WIDTH, fc1_width=FC1_WIDTH, class_count=CLASS_COUNT):
        super(ConvolutionalModel, self).__init__()
        
        # Conv1: 1x28x28 -> 16x28x28 (K=5, P=2)
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=KERNEL_SIZE, 
                               stride=1, padding=2, bias=True)
        # Pool1: 16x28x28 -> 16x14x14 (K=2, S=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2: 16x14x14 -> 32x14x14 (K=5, P=2)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=KERNEL_SIZE, 
                               stride=1, padding=2, bias=True)
        # Pool2: 32x14x14 -> 32x7x7 (K=2, S=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fully connected layers
        #  32 * 7 * 7 = 1568
        self.flattened_size = conv2_width * 7 * 7 
        self.fc1 = nn.Linear(self.flattened_size, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # bias to 0 
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                 # bias to 0 
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        # layer 1: Conv -> ReLU -> Pool
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.pool1(h)
        
        # layer 2: Conv -> ReLU -> Pool
        h = self.conv2(h)
        h = torch.relu(h)
        h = self.pool2(h) # h.shape est (N, 32, 7, 7)
        
        # Flattening
        h = h.view(h.shape[0], -1) 
        
        # layer FC 1 -> ReLU
        h = self.fc1(h)
        h = torch.relu(h)
        
        # out
        logits = self.fc_logits(h)
        return logits
    

def update_learning_rate(optimizer, epoch, lr_policy):
    if epoch in lr_policy:
        new_lr = lr_policy[epoch]['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    return optimizer.param_groups[0]['lr']

def save_filters(model, epoch, save_dir):
    # save filters on first conv layer
    save_dir.mkdir(parents=True, exist_ok=True)
    
    #get w from conv1D
    weights = model.conv1.weight.data.cpu().numpy()
    num_filters = weights.shape[0]
    
    #normalize for visualization
    weights = weights.transpose(2, 3, 1, 0)  # [H, W, in_ch, out_ch]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    # filter grid
    cols = 8
    rows = int(np.ceil(num_filters / cols))
    k = weights.shape[0]
    border = 1
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    
    img = np.ones([height, width, weights.shape[2]])
    
    for i in range(num_filters):
        r = (i // cols) * (k + border)
        c = (i % cols) * (k + border)
        img[r:r+k, c:c+k, :] = weights[:, :, :, i]
    if img.shape[2] == 1:
        img = img[:, :, 0]
    
    # save
    filename = save_dir / f'filters_epoch_{epoch:02d}.png'
    plt.imsave(filename, img, cmap='gray' if len(img.shape) == 2 else None)
    
if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
    SAVE_DIR = Path(__file__).parent / 'out_cifar1O'

    config = {}
    config['max_epochs'] = 8
    config['batch_size'] = 50
    config['save_dir'] = SAVE_DIR
    config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}
    config['weight_decay'] = 0

    def dense_to_one_hot(y, class_count):
        return np.eye(class_count)[y]
    
    def draw_image(img, mean, std):
        img = img.transpose(1, 2, 0)
        img *= std
        img += mean
        img = img.astype(np.uint8)
        ski.io.imshow(img)
        ski.io.show()

    ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
    
    # Conversion en NumPy, normalisation (0-1), et reshape [N, 1, 28, 28]
    train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
    train_y = ds_train.targets.numpy()
    
    # Splitting and Mean Normalization
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
    test_y = ds_test.targets.numpy()
    
    train_mean = train_x.mean()
    train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
    
    # One-Hot Encoding
    train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))
    
    # convert to pytorch tensor
    train_x_pt, valid_x_pt, test_x_pt = map(torch.from_numpy, (train_x, valid_x, test_x))
    train_y_pt, valid_y_pt, test_y_pt = map(torch.from_numpy, (train_y, valid_y, test_y))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvolutionalModel().to(device)
    
    criterion = nn.CrossEntropyLoss()
    #optimizer.SDG
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['lr_policy'][1]['lr'], # LR initial
        weight_decay=config['weight_decay']
    )
    #stock metrics
    history = {'train_loss': [], 'valid_acc': [], 'test_acc': []}

    #training
    for epoch in range(1, config['max_epochs'] + 1):
        #update learning rate
        current_lr = update_learning_rate(optimizer, epoch, config['lr_policy'])

        #train on epoch
        model.train()
        train_loss = 0
        N_train = train_x_pt.shape[0]
        
        #iterate on mini batches
        for i in range(0, N_train, config['batch_size']):
            X_batch = train_x_pt[i:i + config['batch_size']].to(device)
            #indices to compute cross entropy loss
            Y_batch_indices = torch.argmax(train_y_pt[i:i + config['batch_size']], dim=1).to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, Y_batch_indices)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / (N_train / config['batch_size'])
        history['train_loss'].append(avg_train_loss)

        # eval
        model.eval()
        
        def evaluate_set(X, Y_one_hot):
            X_tensor = X.to(device)
            Y_indices = torch.argmax(Y_one_hot, dim=1).to(device)
            
            with torch.no_grad():
                logits = model(X_tensor)
                pred = logits.argmax(dim=1, keepdim=True)
                correct = pred.eq(Y_indices.view_as(pred)).sum().item()
            return correct / Y_indices.size(0)

        valid_acc = evaluate_set(valid_x_pt, valid_y_pt)
        test_acc = evaluate_set(test_x_pt, test_y_pt)
        
        history['valid_acc'].append(valid_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch}/{config['max_epochs']} | LR: {current_lr:.1e} | Loss: {avg_train_loss:.4f} | Valid Acc: {valid_acc:.4f}")
        
        # save filters
        save_filters(model, epoch, SAVE_DIR)

    # final set evaluation
    final_test_acc = evaluate_set(test_x_pt, test_y_pt)
    print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

    #plot 
    epochs_list = range(1, config['max_epochs'] + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_list, history['train_loss'], label='Training Loss')
    plt.plot(epochs_list, history['valid_acc'], label='Validation Accuracy', linestyle='--')
    plt.plot(epochs_list, history['test_acc'], label='Test Accuracy', linestyle=':')
    plt.title("Model Performance Evolution")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVE_DIR / "performance_plot.png")
    plt.show()

    print(f"Performance plot saved to {SAVE_DIR / 'performance_plot.png'}")

    CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
    CIFAR_STD = np.array([0.2023, 0.1994, 0.2010])
    STATS = (CIFAR_MEAN, CIFAR_STD)
    def denormalize_and_format_image(img_tensor_chw, mean, std):
        img = img_tensor_chw.copy()
        img = img * std[:, None, None] + mean[:, None, None]
        img = img.transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        return img

  