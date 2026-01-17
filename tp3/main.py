import numpy as np
import torch
from baseline import BaselineModel, evaluate, train_model
from rnn import RNNModel
from vocabulary import NLPDataset, Vocab,  pad_collate_fn, load_embeddings
from torch.utils.data import DataLoader

if __name__ == "__main__":
    #EX1
    train_dataset = NLPDataset('tp3/data/sst_train_raw.csv')
    text_vocab, label_vocab = train_dataset.build_vocab(max_size=-1, min_freq=1)
    print(f"size of vocab text: {len(text_vocab)}")
    print(f"size of vocab labels: {len(label_vocab)}")
    print("\nEXAMPLES ")
    print(f"Index 0 PAD: {text_vocab.itos[0]}")
    print(f"Index 1 UNK: {text_vocab.itos[1]}")
    print(f"Index 2: {text_vocab.itos[2]}")
    print(f"Index 3: {text_vocab.itos[3]}")

    print("\nTEST ON 1 EXAMPLE")
    text_tokens = train_dataset.texts[3]
    label = train_dataset.labels[3]
    print(f"Text: {text_tokens}")
    print(f"Label: {label}")
    
    text_indices, label_index = train_dataset[3]
    print(f"Numericalized text: {text_indices}")
    print(f"Numericalized label: {label_index}")
    
    print("\nDATA LOADER")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, 
                             collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_loader))
    print(f"Texts shape: {texts.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Lengths: {lengths}")
    
       
    #EX 2
    seed = 7052020
    batch_size = 10
    lr = 1e-4
    num_epochs = 5
    device = 'cpu' # --> no nvdia gpu on my computer :'(
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"Using device: {device}")
    
    print("\nLOADING DATA")
    train_dataset = NLPDataset('tp3/data/sst_train_raw.csv')
    valid_dataset = NLPDataset('tp3/data/sst_valid_raw.csv')
    test_dataset = NLPDataset('tp3/data/sst_test_raw.csv')
    
    #build vocab on only train 
    text_vocab, label_vocab = train_dataset.build_vocab(max_size=-1, min_freq=1)
    valid_dataset.set_vocab(text_vocab, label_vocab)
    test_dataset.set_vocab(text_vocab, label_vocab)
    print(f"Vocabulary size: {len(text_vocab)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print("\nLOADING EMBEDDING GLOVE")
    embedding_layer = load_embeddings(text_vocab, 'tp3/data/sst_glove_6b_300d.txt')
    
    # dataoader for all my dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                             shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=pad_collate_fn)
    
    print("\nCREATE MODEL")
    model = BaselineModel(embedding_layer).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    print("\nTRAINING")
    history = train_model(model, train_loader, valid_loader, optimizer, 
                         criterion, num_epochs=num_epochs, device=device)

    print("\nEVALUATION ON TEST SET")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy = {test_metrics['accuracy']:.3f}")
    print(f"Test F1 = {test_metrics['f1']:.3f}")
    print(f"\nConfusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    #EX 3
    rnn_type = 'LSTM' # RNN or GRU or LSTM
    hidden_dim = 150
    lr = 1e-4
    clip_value = 0.25

    model = RNNModel(
        embedding_layer, 
        rnn_type=rnn_type,
        hidden_dim=hidden_dim,
        num_layers=2,
        bidirectional=False
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    

    print(f"\nTRAINING {rnn_type} MODEL")
    history = train_model(
        model, train_loader, valid_loader, optimizer, criterion, 
        num_epochs=10, device=device, clip_value=clip_value
    )
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nACCURACY ON TEST SET: {test_metrics['accuracy']:.2f}%")