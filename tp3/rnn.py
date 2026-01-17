import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """
    recurrent neural network  : rnn(150) -> rnn(150) -> fc(150, 150) -> ReLU() -> fc(150,1)
    """
    def __init__(self, embedding_layer, rnn_type='GRU', hidden_dim=150, 
                 num_layers=2, bidirectional=False, dropout=0.0):
        super().__init__()
        self.embedding = embedding_layer
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        embedding_dim = embedding_layer.embedding_dim
    
        # select class
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[rnn_type]
        
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True 
        )
        
        # output dim
        # if bidirectionnal output * 2 
        if bidirectional:
            rnn_output_dim = hidden_dim * 2 
        else:
            rnn_output_dim = hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )
    
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        batch_size = x.size(0)
        hidden_dim_final = output.size(2)
    
        last_outputs = torch.zeros(batch_size, hidden_dim_final).to(x.device)
        
        # for each back get indice get length - 1
        if lengths is not None:
            for i in range(batch_size):
                last_word_idx = lengths[i] - 1
                last_outputs[i] = output[i, last_word_idx, :]
        else:
            # take last of the sequence
            last_outputs = output[:, -1, :]
        logits = self.fc(last_outputs)
        return logits