import torch
import torch.nn as nn
import torch.nn.functional as F

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k//2, bias=bias))
        self.add_module('bn', nn.BatchNorm2d(num_maps_out))
        self.add_module('relu', nn.ReLU())

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.net = nn.Sequential(
            _BNReluConv(input_channels, 16),
            nn.MaxPool2d(2),
            _BNReluConv(16, 32),
            nn.MaxPool2d(2),
            _BNReluConv(32, 64),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, emb_size)

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = self.net(img)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def loss(self, anchor, positive, negative, margin=1.0):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        # euclidian dist 
        dist_pos = (a_x - p_x).pow(2).sum(1)
        dist_neg = (a_x - n_x).pow(2).sum(1)
        # Triplet Loss formula
        loss = F.relu(dist_pos - dist_neg + margin)
        return loss.mean()
    
    def save_model(model, path="metric_model.pth"):
        # This saves learned parameters
        torch.save(model.state_dict(), path)
        print(f"Model parameters saved to {path}")

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = img.view(img.size(0), -1) # flatten the image 
        return feats