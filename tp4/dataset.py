from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            # Filter out images with target class equal to remove_class
            # YOUR CODE HERE
            mask = self.targets != remove_class
            self.images = self.images[mask]
            self.targets = self.targets[mask]
            if remove_class in self.classes:
                self.classes.remove(remove_class)

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        # find the class of the current anchor image
        current_class = self.targets[index].item()
        # list of all other classes ( = diff numbers )
        other_classes = []
        for c in self.classes:
            if c != current_class:
                other_classes.append(c)
        negative_class = choice(other_classes)
        possible_indices = self.target2indices[negative_class]
        return choice(possible_indices)

    def _sample_positive(self, index):
        #f ind the class of the current image
        current_class = self.targets[index].item()
        #list all indices of images of the same class
        all_indices_same_class = self.target2indices[current_class]
        candidate_indices = []
        for idx in all_indices_same_class:
            if idx != index:
                candidate_indices.append(idx)
        if len(candidate_indices) == 0:
            return index
        return choice(candidate_indices)

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive_idx = self._sample_positive(index)
            negative_idx = self._sample_negative(index)
            positive = self.images[positive_idx].unsqueeze(0)
            negative = self.images[negative_idx].unsqueeze(0)
            return anchor, positive, negative, target_id

    def __len__(self):
        return len(self.images)