import os

import optuna
from optuna.trial import TrialState
import torch
import argparse
import os
import collections
from sklearn import metrics
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch_geometric.nn as nn
from torch_geometric.nn import ChebConv, SAGEConv, MFConv, GATConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, LayerNorm

from dataset_EEG4 import EEG

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])
root = []
seizure_types = []

DEVICE = torch.device("cpu")
BATCH_SIZE = 128
DIR = os.getcwd()
EPOCHS = 50
BALANCED = False

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    K = trial.suggest_int("K", 2, 10)
    aggre = trial.suggest_categorical("aggre", ["cat", "max", "lstm"])
    out_features = trial.suggest_int("out_features", 40, 80, step=4)
    p = trial.suggest_float("p", 0.1, 0.9, step=0.1)
    
    layers = []
    N_LAYER = 4

    in_features = dataset.num_node_features

    for i in range(N_LAYER):
        first_desc = 'x'+str(i)+', edge_index, edge_weight -> x'+str(i)
        sec_desc = 'x'+str(i)+', batch -> x'+str(i+1)
        layers.append((ChebConv(in_features, out_features, K), first_desc))
        layers.append((LayerNorm(out_features), sec_desc))
        layers.append(Dropout(p))
        layers.append(ReLU(inplace=True))
        in_features = out_features  

    layers.append((lambda x1, x2, x3, x4: [x1, x2, x3, x4], 'x1, x2, x3, x4 -> xs'))
    layers.append((JumpingKnowledge(aggre, in_features, num_layers=N_LAYER), 'xs -> x'))
    if aggre != "cat":
        N_LAYER = 1

    # Add pooling, dropout and linear layers as the final classifier
    layers.append((global_mean_pool, 'x, batch -> x'))
    # layers.append((lambda xs : print(xs.shape), 'x -> x'))

    layers.append((Linear(N_LAYER*in_features, N_LAYER*in_features), 'x -> x'))
    layers.append(ReLU(inplace=True))
    layers.append((Dropout(p), 'x -> x'))
    layers.append(Linear(N_LAYER*in_features, dataset.num_classes))
   
    return nn.Sequential('x0, edge_index, edge_weight, batch', [*layers])

# Train classifier on train data
def train(model, loader, criterion, optimizer):
    model.train()
    for data in loader:  
        out = model(data.x, data.edge_index, data.edge_weight, batch=data.batch)
        loss = criterion(out, data.y) 
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad()

# Return the predicted class from all the loader
def get_prediction(model, loader):
    model.eval()
    for step, data in enumerate(loader):
        if step == 0:
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)  
            pred = out.argmax(dim=1)
            true = data.y
        else:
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)  
            pred = torch.cat((pred, out.argmax(dim=1)))
            true = torch.cat((true, data.y))
    return true, pred

def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)
    model = model.double()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = 0.00045 #trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss(weight = dataset.weight)
    
    train_dataset = dataset[dataset.train_mask]
    test_dataset = dataset[~dataset.train_mask]
    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()

    train_data = train_dataset[:8000]
    val_data = train_dataset[8000:11000]
    test_data = test_dataset[:3000]
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Training of the model.
    for epoch in range(EPOCHS):
        train(model, train_loader, criterion, optimizer)

        # Validation of the model.
        y_true, y_pred = get_prediction(model, val_loader)
        f1 = metrics.f1_score(y_true, y_pred,average='weighted')
        trial.report(f1, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    y_true, y_pred = get_prediction(model, val_loader)
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    return f1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create graph and GNN from EEG samples')
    parser.add_argument('--data_dir', default='/home/alclivaz/data/v1.5.2', help='path to the dataset')
    parser.add_argument('--seizure_types',default=['FNSZ','GNSZ','BG','ABSZ','CPSZ','SPSZ','TCSZ','TNSZ'], help="types of seizures")

    args = parser.parse_args()
    seizure_types = args.seizure_types
    data_dir = args.data_dir
    root = data_dir
    os.makedirs(root + "/processed", exist_ok=True)
    dataset = EEG(root, seizure_types, 'False', 'FFT')

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
