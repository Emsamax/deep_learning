import numpy as np
import torch
from baseline import BaselineModel, evaluate, train_model
from rnn import RNNModel
from vocabulary import NLPDataset, Vocab,  pad_collate_fn, load_embeddings
from torch.utils.data import DataLoader


def train_with_special_config(config, train_loader, valid_loader, test_loader, embedding_layer, device):
    """train on a given configuration and return the accuracy"""
    print(f"\n>>> Testing Config: {config}")
    
    if config['model_type'] == 'baseline':
        model = BaselineModel(embedding_layer, hidden_dim=config['hidden_dim']).to(device)
    else:
        model = RNNModel(
            embedding_layer, 
            rnn_type=config['rnn_type'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            bidirectional=config['bidirectional'],
            dropout=config['dropout']
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.BCEWithLogitsLoss()
    
    history = train_model(
        model, train_loader, valid_loader, optimizer, criterion, 
        num_epochs=config['epochs'], device=device, clip_value=config['clip']
    )
    
    final_val_acc = history['valid_accuracy'][-1]
    return final_val_acc

if __name__ == "__main__":
    #EX1
    print("="*40)
    print("EX 1")
    print("="*40)
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
    print("="*40)
    print("EX 2")
    print("="*40)
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
    
    #EX 3/4
    print("="*40)
    print("EX 3 and 4 (hyperparameters exploration)")
    print("="*40)
    
    #save the hyperparameters  of the config with the best accuracy
    best_config = {
        'accuracy': -1.0,
        'params': None
    }
    
    rnn_types = ['RNN', 'GRU', 'LSTM']
    hidden_sizes = [150, 300, 500]
    num_layers_list = [1, 2, 3]
    bidirectionals = [True, False]
    
    results = []

    print("\nSTARTING HYPERPARAMETER EXPLORATION FOR OPTIMIZATION")
    number = 1
    for r_type in rnn_types:
        for h_size in hidden_sizes:
            current_config = {
                'model_type': 'rnn',
                'rnn_type': r_type,
                'hidden_dim': h_size,
                'num_layers': 2,
                'bidirectional': True,
                'dropout': 0.3,
                'lr': 1e-4,
                'epochs': 5,
                'clip': 0.25
            }
      
            print(f"CONFIG NUMBER: {number:.0f}")
            number += 1
            current_acc = train_with_special_config(current_config, train_loader, valid_loader, test_loader, embedding_layer, device)
            
            # update the best model
            if current_acc > best_config['accuracy']:
                print(f" Better config found with accuracy of : {current_acc:.2f}% (Old: {best_config['accuracy']:.2f}%)")
                best_config['accuracy'] = current_acc
                best_config['params'] = current_config

    print("\nRESULTS")
    for res in results:
        print(f"Type: {res[0]}, Hidden: {res[1]}, Accuracy: {res[2]:.2f}%")

    seed_results = []
    final_params = best_config['params']
    print(f"\n=== TESTING BEST CONFIG WITH 5 SEEDS ")
    for s in [42, 100, 500, 1000, 10000]:
        torch.manual_seed(s)
        np.random.seed(s)
        acc = train_with_special_config(final_params, train_loader, valid_loader, test_loader, embedding_layer, device)
        seed_results.append(acc)
    
    print(f"\nFinal Statistics for Best Model: Mean = {np.mean(seed_results):.2f}%, Std = {np.std(seed_results):.2f}%")
