import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np


class Vocab:
    def __init__(self, frequencies: Dict[str, int], max_size: int = -1, 
                 min_freq: int = 1, special_symbols: bool = True):
        self.special_symbols = special_symbols
        self.itos = []  # index to string
        self.stoi = {}  # string to index
        if special_symbols:
            self.itos.append('<PAD>')
            self.itos.append('<UNK>')
        # sort by desc frequencies
        filtered_freqs = {word: freq for word, freq in frequencies.items() 
                         if freq >= min_freq}
        sorted_words = sorted(filtered_freqs.items(), key=lambda x: x[1], reverse=True)
        if max_size > 0:
            num_special = 2 if special_symbols else 0
            sorted_words = sorted_words[:max_size - num_special]
        # build vocabulary
        for word, _ in sorted_words:
            self.itos.append(word)
        # reverse mapping 
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}
    
    def encode(self, text):
        if isinstance(text, str):
            # 1 token
            idx = self.stoi.get(text, self.stoi.get('<UNK>', 1))
            return torch.tensor(idx)
        else:
            # list of tokens
            indices = []
            for token in text:
                if self.special_symbols:
                    idx = self.stoi.get(token, self.stoi['<UNK>'])
                else:
                    idx = self.stoi.get(token, 0)
                indices.append(idx)
            return torch.tensor(indices)
    
    def __len__(self):
        return len(self.itos)


class NLPDataset(Dataset):
    def __init__(self, filepath: str, text_vocab: Optional[Vocab] = None,
                 label_vocab: Optional[Vocab] = None):
        self.texts = []
        self.labels = []
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab
        self._load_from_file(filepath)
    
    def _load_from_file(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(', ', 1)
                if len(parts) != 2:
                    continue
                text_str, label = parts
                text_tokens = text_str.split()
                self.texts.append(text_tokens)
                self.labels.append(label)
    
    def build_vocab(self, max_size: int = -1, min_freq: int = 1):
        text_frequencies = Counter()
        label_frequencies = Counter()
        
        for text_tokens in self.texts:
            text_frequencies.update(text_tokens)
        
        for label in self.labels:
            label_frequencies[label] += 1
            
        self.text_vocab = Vocab(text_frequencies, max_size=max_size, 
                               min_freq=min_freq, special_symbols=True)
        self.label_vocab = Vocab(label_frequencies, max_size=-1, 
                                min_freq=0, special_symbols=False)
        
        return self.text_vocab, self.label_vocab
    
    def set_vocab(self, text_vocab: Vocab, label_vocab: Vocab):
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text_tokens = self.texts[idx]
        label = self.labels[idx]
        text_indices = self.text_vocab.encode(text_tokens)
        label_index = self.label_vocab.encode(label)
        return text_indices, label_index


def pad_collate_fn(batch, pad_index=0):
    """
    create batches with padding 
    Args:
        batch: list of tuples (text_indices, label_index)
        pad_index: index of token padding 
    Returns:
        texts: Tensor [batch_size, max_seq_len]
        labels: Tensor [batch_size]
        lengths: Tensor [batch_size] 
    """
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, 
                                             padding_value=pad_index)
    labels = torch.stack(labels)
    return texts_padded, labels, lengths


def load_embeddings(vocab: Vocab, embedding_file: str, 
                   embedding_dim: int = 300) -> nn.Embedding:
    """
    load glove pretrained vectors
    """
   #init with normal dostrib
    embeddings = np.random.randn(len(vocab), embedding_dim).astype(np.float32)
    embeddings[0] = np.zeros(embedding_dim)
    found = 0
    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != embedding_dim + 1:
                    continue
                word = parts[0]
                if word in vocab.stoi:
                    idx = vocab.stoi[word]
                    vector = np.array([float(x) for x in parts[1:]])
                    embeddings[idx] = vector
                    found += 1
        
        print(f"Loaded {found}/{len(vocab)} embeddings from file")
    except FileNotFoundError:
        print(f"Warning: Embedding file {embedding_file} not found. Using random initialization.")
    
    embedding_layer = nn.Embedding.from_pretrained(
        torch.FloatTensor(embeddings),
        padding_idx=0,
        freeze=True  # True for pretrained false else
    )
    return embedding_layer


