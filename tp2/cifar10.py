import torch
from torch import nn
import torch.nn.functional as F # Utiliser F.relu pour rester clair
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR # Ajout du scheduler PyTorch

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

import os
import numpy as np
import matplotlib.pyplot as plt 
import skimage.io as ski_io
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path(__file__).parent / 'datasets'
SAVE_DIR = Path(__file__).parent / 'out_cifar10'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10 
LR_POLICY = {1: 1e-1, 5: 1e-2, 8: 1e-3} 
INITIAL_LR = LR_POLICY[1]


def update_learning_rate(optimizer, epoch, lr_policy):
    if epoch in lr_policy:
        lr = lr_policy[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Setting learning rate to {lr:.1e} at epoch {epoch}")
        return lr
    return optimizer.param_groups[0]['lr']

def draw_conv_filters(epoch, step, weights, save_dir):
    """save convolution filters"""
    w = weights.copy()
    num_filters, num_channels, k = w.shape[:3]
    w = w.transpose(2, 3, 1, 0) # (K, K, C, N)
    w -= w.min()
    w /= w.max() if w.max() > 0 else 1.0 # Normalisation min-max
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k, c:c+k, :] = w[:, :, :, i]

    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski_io.imsave(os.path.join(save_dir, filename), (img * 255).astype(np.uint8))

def evaluate(model, data_loader, device, criterion, dataset_name="Dataset"):
   #loss on dataset
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())

    N = len(data_loader.dataset)
    avg_loss = total_loss / N
    accuracy = accuracy_score(all_targets, all_preds)
    
    precision, recall, _, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(NUM_CLASSES), average=None, zero_division=0
    )
    conf_mat = confusion_matrix(all_targets, all_preds)

    print(f"\n--- Evaluation on {dataset_name} ---")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print("Metrics per class:")
    print("Class \t Precision \t Recall")
    for i in range(NUM_CLASSES):
        print(f"{i} \t {precision[i]:.4f} \t\t {recall[i]:.4f}")
    
    return avg_loss, accuracy

def plot_training_progress(save_dir, data, lr_list):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    x_data = np.linspace(1, len(data['train_loss']), len(data['train_loss']))
    
    #loss
    ax1.set_title('Cross-entropy loss'); ax1.plot(x_data, data['train_loss'], label='train')
    ax1.plot(x_data, data['valid_loss'], label='validation'); ax1.legend(loc='upper right')
    
    #precision
    ax2.set_title('Average class accuracy'); ax2.plot(x_data, data['train_acc'], label='train')
    ax2.plot(x_data, data['valid_acc'], label='validation'); ax2.legend(loc='lower right')
    
    #learning rate
    ax3.set_title('Learning rate'); ax3.plot(x_data, lr_list, label='learning_rate'); ax3.legend(loc='upper left')

    ax4.axis('off') 

    save_path = os.path.join(save_dir, 'training_plot.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)
    plt.close(fig) 

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #define model layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fc_in_size = 32 * 7 * 7
        self.fc1 = nn.Linear(self.fc_in_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_logits = nn.Linear(128, NUM_CLASSES)
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and m is not self.fc_logits):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x); h = F.relu(h); h = self.pool1(h)
        h = self.conv2(h); h = F.relu(h); h = self.pool2(h)
        h = h.view(h.size(0), -1) 
        h = self.fc1(h); h = F.relu(h)
        h = self.fc2(h); h = F.relu(h)
        logits = self.fc_logits(h)
        return logits

if __name__ == "__main__":
    
    def denormalize_and_format_image(img_tensor_chw, mean, std):
      img = img_tensor_chw.copy()
      mean_reshaped = mean[:, None, None]
      std_reshaped = std[:, None, None]
      # x * std + mean
      img = img * std_reshaped + mean_reshaped
      img = img.transpose(1, 2, 0)
      img = np.clip(img, 0, 1)
      return img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #load cifar data
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    train_val_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

   
   
    train_size = len(train_val_dataset) - 10000
    valid_size = 10000
    train_dataset, valid_dataset = random_split(train_val_dataset, [train_size, valid_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Instanciation
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR,)

    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'lr': []}
    #filters
    draw_conv_filters(epoch=0, step=0, weights=model.conv1.weight.detach().cpu().numpy(), save_dir=SAVE_DIR)
    #training
    for epoch in range(1, NUM_EPOCHS + 1):
        
        # update learning rate
        current_lr = update_learning_rate(optimizer, epoch, LR_POLICY)
        history['lr'].append(current_lr)
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            if batch_idx % 200 == 0:
                 draw_conv_filters(epoch, batch_idx, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)

        #evaluate on training set
        train_loss, train_acc = evaluate(model, train_loader, device, criterion, dataset_name="Train Set")
        #evaluate on validation set
        valid_loss, valid_acc = evaluate(model, valid_loader, device, criterion, dataset_name="Validation Set")

        #save loss and accuracy 
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")

    plot_training_progress(SAVE_DIR, history, history['lr'])
    print("\nfinal evaluation")
    evaluate(model, test_loader, device, criterion, dataset_name="Test Set")

    print("\n20 worst misclassified images")

    # 1 all test dataset in 1 tensor
    all_test_data = []
    all_test_targets = []
    for data, target in test_loader:
        all_test_data.append(data)
        all_test_targets.append(target)

    test_x_pt = torch.cat(all_test_data)
    test_y_indices = torch.cat(all_test_targets)
    
    CIFAR_MEAN_NP = np.array(CIFAR_MEAN)
    CIFAR_STD_NP = np.array(CIFAR_STD)

    model.eval()
    criterion_individual = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        logits = model(test_x_pt.to(device))
        #compute all losses
        losses = criterion_individual(logits, test_y_indices.to(device)).cpu().numpy()
        
        # predictions
        predictions = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    
    #Identify 20 worst
    true_labels = test_y_indices.numpy()
    misclassified_mask = predictions != true_labels
    misclassified_indices = np.where(misclassified_mask)[0]
    
    #sort desc
    misclassified_losses = losses[misclassified_indices]
    sorted_idx = np.argsort(misclassified_losses)[::-1]
    worst_20_indices = misclassified_indices[sorted_idx[:20]]
    
    test_x_raw = test_x_pt.cpu().numpy() # Images normalisées (N, C, H, W)
    
    #plot in a grid
    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    fig.suptitle('20 Worst Misclassified Images (Largest Loss)', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    
    CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for idx, ax in enumerate(axes.flat):
        if idx >= len(worst_20_indices):
            ax.axis('off')
            continue
        img_idx = worst_20_indices[idx]
        img_data_chw = test_x_raw[img_idx]
        img_display = denormalize_and_format_image(img_data_chw, CIFAR_MEAN_NP, CIFAR_STD_NP)
        correct_class = true_labels[img_idx]
        predicted_class = predictions[img_idx]
        #3 best pred for this img
        top3_indices = np.argsort(probs[img_idx])[::-1][:3]
        top3_probs = probs[img_idx][top3_indices]
        top3_classes = [CIFAR_CLASSES[i] for i in top3_indices]
        ax.imshow(img_display)
        ax.axis('off')
        title_text = (f"Cor: {CIFAR_CLASSES[correct_class]}\n"f"Pred: {top3_classes[0]} ({top3_probs[0]:.2f})\n"f"Loss: {losses[img_idx]:.3f}")
        ax.set_title(title_text, fontsize=9, color='red') 
    
    # save and plot
    save_path = SAVE_DIR / 'worst_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nWorst predictions saved to {save_path}")
    print(f"Total misclassified in test set: {len(misclassified_indices)}")
    print(f"Worst loss encountered: {losses[worst_20_indices[0]]:.4f}")
    print("\nfinal evaluation")
    evaluate(model, test_loader, device, criterion, dataset_name="Test Set")