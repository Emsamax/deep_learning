import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

class BaselineModel(nn.Module):
    """
    Model baseline with average pooling : avg_pool() -> fc(300, 150) -> ReLU() -> fc(150, 150) -> ReLU() -> fc(150, 1)
    """
    
    def __init__(self, embedding_layer, hidden_dim=150):
        super().__init__()
        self.embedding = embedding_layer
        embedding_dim = embedding_layer.embedding_dim
    
        # fully connected layers as describe above
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: [batch_size, seq_len] token indices
            lengths: [batch_size] length of sequences
        Returns:
            logits: [batch_size, 1]
        """
        embedded = self.embedding(x)
        if lengths is not None:
            # mask to ignore padding
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float() 
            #applies mask and computes average
            masked_embedded = embedded * mask
            pooled = masked_embedded.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            # no mask
            pooled = embedded.mean(dim=1)
        
        # Forward 
        x = self.relu(self.fc1(pooled))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device='cpu', clip_value=None):
    """
    train for 1 epoch and retrun loss for this epoch
    """
    model.train()
    total_loss = 0
    for batch_num, (texts, labels, lengths) in enumerate(dataloader):
        texts = texts.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        lengths = lengths.to(device)
        # Forward
        optimizer.zero_grad()
        logits = model(texts, lengths)
        loss = criterion(logits, labels)
        # Backward
        loss.backward()
        
        # Gradient clipping (optionnel)
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, criterion, device='cpu'):
    """
    Evaluates model
    
    Returns:
        metrics: Dict [loss, accuracy, f1, confusion_matrix]
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts = texts.to(device)
            labels_float = labels.to(device).float().unsqueeze(1)
            lengths = lengths.to(device)
            
            # Forward
            logits = model(texts, lengths)
            loss = criterion(logits, labels_float)
            
            # predict
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().view(-1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='binary') * 100
    conf_matrix = confusion_matrix(all_labels, all_preds)
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }
    return metrics


def train_model(model, train_loader, valid_loader, optimizer, criterion, 
                num_epochs=5, device='cpu', clip_value=None):
    """
    train molel
    
    Returns:
        history: Dict[metrics]
    """
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_accuracy': [],
        'valid_f1': []
    }
    
    for epoch in range(num_epochs):
        # train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                device, clip_value)
        
        # evaluate
        valid_metrics = evaluate(model, valid_loader, criterion, device)
        
        # save in history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_metrics['loss'])
        history['valid_accuracy'].append(valid_metrics['accuracy'])
        history['valid_f1'].append(valid_metrics['f1'])
        
        # print results
        print(f"Epoch {epoch + 1}: "
              f"train_loss = {train_loss:.4f}, "
              f"valid_loss = {valid_metrics['loss']:.4f}, "
              f"valid_accuracy = {valid_metrics['accuracy']:.3f}, "
              f"valid_f1 = {valid_metrics['f1']:.3f}")
    return history