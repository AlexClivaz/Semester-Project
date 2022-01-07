from random import shuffle
import torch
import argparse
import os
import collections
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn import Linear, Dropout, ReLU
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch_geometric.nn as nn
from torch_geometric.nn import JumpingKnowledge, GNNExplainer, SAGEConv
from torch_geometric.nn import global_mean_pool, LayerNorm
from captum.attr import IntegratedGradients
from dataset_EEGexp import EEG

np.set_printoptions(precision=4, suppress=True)

TRAIN = False
SAVE = True
BATCH_SIZE = 64
DROPOUT = 0.5
EPOCH = 5
LR = 0.0007
HIDDEN_CHANNELS = 80
NB_LAYERS = 4
K = 5
model_name = 'Explanation'
# model_name = 'GCN_'+str(NB_LAYERS)+'_'+str(HIDDEN_CHANNELS)

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

# Create a model using sequential module of pytorch
def define_model():
    layers = []

    # Make two copies of the normalized data
    in_features = dataset.num_node_features
    p = DROPOUT
    
    layers.append((lambda x: x, 'x -> x0'))
    # Add convolutional layers
    for i in range(NB_LAYERS):
        first_desc = 'x'+str(i)+', edge_index -> x'+str(i)
        sec_desc = 'x'+str(i)+', batch -> x'+str(i+1)
        layers.append((SAGEConv(in_features, HIDDEN_CHANNELS), first_desc))
        layers.append((LayerNorm(HIDDEN_CHANNELS), sec_desc))
        layers.append(Dropout(p))
        layers.append(ReLU(inplace=True))
        in_features = HIDDEN_CHANNELS
    
    # Merge the data
    layers.append((lambda x1, x2, x3, x4: [x1, x2, x3, x4], 'x1, x2, x3, x4 -> xs'))
    layers.append((JumpingKnowledge("max", in_features, num_layers=NB_LAYERS), 'xs -> x'))
    
    # Add pooling, dropout and linear layers as the final classifier
    layers.append((global_mean_pool, 'x, batch -> x'))
    layers.append((Linear(in_features, NB_LAYERS*in_features), 'x -> x'))
    layers.append(ReLU(inplace=True))
    layers.append(Dropout(p))

    layers.append(Linear(NB_LAYERS*in_features, dataset.num_classes))
   
    return nn.Sequential('x, edge_index, batch', [*layers])

# Train classifier on train data
def train(model):
    model.train()
    t_loss = 0
    for data in train_loader:  
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y) 
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
        t_loss += loss.item()
    return t_loss/len(train_loader)

# Test classifier on data and return accuracy
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader: 
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)  
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y).sum()) 
    return correct / len(loader.dataset)

# Return the predicted class from all the loader
def get_prediction(model, loader):
    model.eval()
    for step, data in enumerate(loader):
        if step == 0:
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)
            true = data.y
        else:
            out = model(data.x, data.edge_index, data.batch)  
            pred = torch.cat((pred, out.argmax(dim=1)))
            true = torch.cat((true, data.y))
    return true, pred

# Define forward function for model explanation
def model_forward(node_mask, data):
    batch = torch.zeros(node_mask.shape[0]*data.x.shape[0], dtype=int)
    edge_index = data.edge_index.detach().repeat(1,node_mask.shape[0])
    x_mask = torch.cat([(node_mask[i,:]*data.x.t()).t() for i in range(node_mask.shape[0])], 0)
    out = model(x_mask, edge_index, batch)
    return out

# Return node mask containing node importance
def explain(ind):
    data = dataset[ind]
    node_mask_mean = np.zeros((1,data.x.shape[0]))
    for i in range(5):
        input_mask = torch.ones(data.x.shape[0]).unsqueeze(0).requires_grad_(True)
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask,target=0,additional_forward_args=(data,), n_steps=100, internal_batch_size=data.x.shape[0])
        node_mask = np.abs(mask.cpu().detach().numpy())
        if node_mask.max() > 0:  # avoid division by zero
            node_mask = node_mask / node_mask.max()
        node_mask_mean += node_mask
    node_mask_mean /= 5
    return node_mask_mean.squeeze()

# Get explanation from GNN Explainer
def gnn_explain(ind):
    explainer = GNNExplainer(model, epochs=200, return_type='log_prob', feat_mask_type = 'scalar', log=False)
    data = dataset[ind]
    node_mask_mean = np.zeros((1,data.x.shape[0]))
    for i in range(5):
        x, edge_index = data.x, data.edge_index
        node_mask, edge_mask = explainer.explain_graph(x, edge_index)
        node_mask = node_mask.detach().numpy()
        node_mask = node_mask/node_mask.max()
        node_mask_mean += node_mask
    node_mask_mean = node_mask_mean/5
    return node_mask_mean.squeeze()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create graph and GNN from EEG samples')
    parser.add_argument('--data_dir', default='./data/v1.5.2', help='path to the dataset')
    parser.add_argument('--seizure_types',default=['FNSZ','GNSZ','ABSZ','CPSZ','TNSZ'], help="types of seizures")
    parser.add_argument('--dataset_args',nargs="*",default=['False','FFT','Dist', '12'], help="dataset characteristics")
    args = parser.parse_args()
    seizure_types = args.seizure_types
    data_dir = args.data_dir
    dataset_args = args.dataset_args

    root = data_dir
    os.makedirs(root + "/processed", exist_ok=True)
    dataset = EEG(root, seizure_types, dataset_args[0], dataset_args[1], dataset_args[2], dataset_args[3])
    print(dataset)
    print(dataset[0].x.shape)

    # Split dataset in training and testing
    train_dataset = dataset[dataset.train_mask]
    test_dataset = dataset[~dataset.train_mask]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    k_fold = 3
    conf_matrix = []
    loss_matrix = []
    f1_matrix = []
    f1_weight = []
    f1_macro = []
    conf_val = []
    f1_val = []

    if TRAIN == False:
        model = define_model().double()
        model.load_state_dict(torch.load('./model/'+model_name+'.pt'))
        model.eval()
        ind = (np.random.rand((10))*len(dataset)).astype(int)
        for iter,index in enumerate(ind):
            data = dataset[index]
            expl = explain(index)
            gnn_expl = gnn_explain(index)
            energy = torch.sum(dataset[index].x[:,5:], dim=1)
            energy -= energy.min()
            energy /= energy.max()
            var = dataset[index].x[:,4]
            var /= var.max()
            batch = torch.zeros(data.x.shape[0], dtype=int)
            out = model(data.x, data.edge_index, batch)
            print(out, data.y)

            x = np.arange(19)  # the label locations
            width = 0.2  # the width of the bars
            fig1, ax1 = plt.subplots()
            rects1 = ax1.bar(x - width, expl, width, label='Expl')
            rects2 = ax1.bar(x, gnn_expl, width, label='GNNEplx')
            rects3 = ax1.bar(x + width, energy, width, label='Energy')
            # rects3 = ax.bar(x + 3/2*width, var, width, label='Variance')

            ax1.set_ylabel('Normalized value')
            ax1.set_xlabel('Node')
            ax1.set_xticks(x)
            ax1.legend()
            fig1.tight_layout()

            fig2, ax2 = plt.subplots()
            time_serie =  data.time_serie.T
            time_serie = time_serie - time_serie[0,:] + np.arange(19)
            plt.plot(np.arange(0,2,0.004), time_serie)
            ax2.set_ylabel('Node')
            ax2.set_yticks(np.arange(19))
            # ax2.set_xticks(np.arange(0,2,0.004))
            ax2.set_xlabel('Time [s]')
            fig2.tight_layout()
            plt.show()
        exit()
    
    val_rand = torch.randperm(len(dataset))
    for i in range(1):
        printstr = "Repetition number: " + str(i+1) + "  over " + str(k_fold)
        print(printstr)

        # Create different dataset, train, val
        samples_weight = dataset.weight[dataset.data.y[dataset.train_mask]]
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
            
        # Define model, criterion and optimizer
        model = define_model().double()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        l_loss = []
        l_f1 = []

        nb_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of parameters: {nb_param}')

        # Train model on train dataset and evaluate on the validation dataset
        # (Since the same patients are both in train and validation, results are far
        # better than on test dataset)
        for epoch in range(EPOCH):
            loss = train(model)
            y_true, y_pred = get_prediction(model, test_loader)
            f1_score = metrics.f1_score(y_true, y_pred,average='weighted')
            f1_MACRO = metrics.f1_score(y_true, y_pred,average='macro')
            print(f'Epoch: {epoch:03d}, f1 weighted: {f1_score}, f1 macro: {f1_MACRO}')
            l_loss.append(loss)
            l_f1.append(f1_score)

        # Save model weights
        if SAVE:
            torch.save(model.state_dict(), './model/'+model_name+'.pt')

        conf_val.append(metrics.confusion_matrix(y_true, y_pred, normalize='true'))
        f1_val.append(metrics.f1_score(y_true, y_pred,average='weighted'))

        # Evaluate model on test dataset, metrics are f1 scores
        y_true, y_pred = get_prediction(model, test_loader)
        conf_matrix.append(metrics.confusion_matrix(y_true, y_pred, normalize='true'))
        f1_weight.append(metrics.f1_score(y_true, y_pred,average='weighted'))
        f1_macro.append(metrics.f1_score(y_true, y_pred,average='macro'))
        loss_matrix.append(l_loss)
        f1_matrix.append(l_f1)
        print(metrics.classification_report(y_true, y_pred, target_names=seizure_types))
        print("\n")

    # Print final results
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(np.mean(loss_matrix, axis=0))
    # plt.subplot(212)
    # plt.plot(np.mean(f1_matrix, axis=0))
    # plt.show()

    # Save results
    file = open("./data/model/"+model_name+".txt","a")
    file.write("Final score (mean) on validation"+"\n\n""Confusion Matrix\n")
    file.write(str(seizure_types)+"\n")
    file.write(str(np.mean(conf_val, axis=0)))
    file.write("\n\nF1 score weighted\n"+str(np.mean(f1_val)))
    file.write("\n\nFinal score (mean) on test"+"\n\n""Confusion Matrix\n")
    file.write(str(seizure_types)+"\n")
    file.write(str(np.mean(conf_matrix, axis=0)))
    file.write("\n\nF1 score weighted\n"+str(np.mean(f1_weight)))
    file.write("\nF1 score macro\n"+str(np.mean(f1_macro))+"\n\n")
    file.close()
