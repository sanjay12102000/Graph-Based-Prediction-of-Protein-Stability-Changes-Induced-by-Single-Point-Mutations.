import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load the list of wild types, mutant types, and labels
data_path = "/media/mahesh/49e17af6-9b59-4c4d-ae39-c3d26543ffec/Mahesh/sanjay/sample_output.csv"
data = pd.read_csv(data_path)

def load_graph_data(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)

def create_adjacency_matrix(edges, num_nodes):
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        adjacency_matrix[source, target] = 1
    return adjacency_matrix

def create_feature_matrix(nodes):
    num_features = 9  # Number of features per node
    feature_matrix = np.zeros((len(nodes), num_features))

    for i, node in enumerate(nodes):
        feature_matrix[i, 0] = node["hydrophobicity"]
        feature_matrix[i, 1] = node["charge"]
        feature_matrix[i, 2] = node["molecular_weight"]
        feature_matrix[i, 3] = node["hydrophobicity_Hh"]
        feature_matrix[i, 4] = node["VSc"]
        feature_matrix[i, 5] = node["p1"]
        feature_matrix[i, 6] = node["p2"]
        feature_matrix[i, 7] = node["SASA"]
        feature_matrix[i, 8] = node["NCISC"]

    return feature_matrix

def process_graph(file_path):
    graph_data = load_graph_data(file_path)
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    num_nodes = len(nodes)
    adjacency_matrix = create_adjacency_matrix(edges, num_nodes)
    feature_matrix = create_feature_matrix(nodes)
    return adjacency_matrix, feature_matrix

# Directories containing JSON files
wild_directory = "/media/mahesh/49e17af6-9b59-4c4d-ae39-c3d26543ffec/Mahesh/sanjay/new_graphs/wild_graphs/"
mutant_directory = "/media/mahesh/49e17af6-9b59-4c4d-ae39-c3d26543ffec/Mahesh/sanjay/new_graphs/"

graph_pairs = []

for index, row in data.iterrows():
    wild_file = os.path.join(wild_directory, row['wildtype_graph'])
    mutant_file = os.path.join(mutant_directory, row['Mutated graphs'])
    label = row['ddG_ML']
    
    wild_adj, wild_features = process_graph(wild_file)
    mutant_adj, mutant_features = process_graph(mutant_file)
    
    graph_pairs.append((wild_adj, wild_features, mutant_adj, mutant_features, label))

def to_pyg_data(adjacency_matrix, feature_matrix, label):
    edge_index = np.array(adjacency_matrix.nonzero())
    x = torch.tensor(feature_matrix, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

pyg_data_list = []

for wild_adj, wild_features, mutant_adj, mutant_features, label in graph_pairs:
    wild_data = to_pyg_data(wild_adj, wild_features, label)
    mutant_data = to_pyg_data(mutant_adj, mutant_features, label)
    pyg_data_list.append((wild_data, mutant_data))

class GATv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def save_checkpoint(epoch, model, optimizer, fold, path='/media/mahesh/49e17af6-9b59-4c4d-ae39-c3d26543ffec/Mahesh/sanjay'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'fold': fold
    }, path)

def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        fold = checkpoint['fold']
        return model, optimizer, start_epoch, fold
    else:
        return model, optimizer, 0, 0

def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_targets = []
    all_predictions = []
    for wild_data, mutant_data in data_loader:
        optimizer.zero_grad()
        wild_out = model(wild_data.x, wild_data.edge_index).squeeze()
        mutant_out = model(mutant_data.x, mutant_data.edge_index).squeeze()
        loss = criterion(wild_out, mutant_out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_targets.extend(mutant_out.detach().cpu().numpy())
        all_predictions.extend(wild_out.detach().cpu().numpy())

    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    pearson_corr, _ = pearsonr(all_targets, all_predictions)
    
    accuracy = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)) < 0.1)
    
    return total_loss / len(data_loader), mse, r2, pearson_corr, accuracy, all_targets, all_predictions

def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for wild_data, mutant_data in data_loader:
            wild_out = model(wild_data.x, wild_data.edge_index).squeeze()
            mutant_out = model(mutant_data.x, mutant_data.edge_index).squeeze()
            loss = criterion(wild_out, mutant_out)
            total_loss += loss.item()

            all_targets.extend(mutant_out.detach().cpu().numpy())
            all_predictions.extend(wild_out.detach().cpu().numpy())

    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    pearson_corr, _ = pearsonr(all_targets, all_predictions)
    
    accuracy = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)) < 0.1)
    
    return total_loss / len(data_loader), mse, r2, pearson_corr, accuracy, all_targets, all_predictions

def run_k_fold_cross_validation(pyg_data_list, k=10, epochs=150, batch_size=32, in_channels=9, out_channels=1, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    all_train_r2s = []
    all_val_r2s = []
    all_train_pearson_corrs = []
    all_val_pearson_corrs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(pyg_data_list)):
        print(f"Fold {fold + 1}/{k}")

        train_data = [pyg_data_list[i] for i in train_idx]
        val_data = [pyg_data_list[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        model = GATv2(in_channels, out_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.MSELoss()

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_fold_{fold+1}.pth')
        model, optimizer, start_epoch, fold_loaded = load_checkpoint(checkpoint_path, model, optimizer)
        
        if fold_loaded != fold:
            start_epoch = 0  # Restart training if resuming with a different fold

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_r2s = []
        val_r2s = []
        train_pearson_corrs = []
        val_pearson_corrs = []

        for epoch in range(start_epoch, epochs):
            train_loss, _, train_r2, train_pearson_corr, train_accuracy, _, _ = train(model, train_loader, optimizer, criterion)
            val_loss, val_mse, val_r2, val_pearson_corr, val_accuracy, _, _ = validate(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_r2s.append(train_r2)
            val_r2s.append(val_r2)
            train_pearson_corrs.append(train_pearson_corr)
            val_pearson_corrs.append(val_pearson_corr)

            save_checkpoint(epoch + 1, model, optimizer, fold, checkpoint_path)  # Save checkpoint after each epoch

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accuracies.append(train_accuracies)
        all_val_accuracies.append(val_accuracies)
        all_train_r2s.append(train_r2s)
        all_val_r2s.append(val_r2s)
        all_train_pearson_corrs.append(train_pearson_corrs)
        all_val_pearson_corrs.append(val_pearson_corrs)

        # Save the final model for the current fold
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'gatv2_fold_{fold+1}.pth'))

    return all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, all_train_r2s, all_val_r2s, all_train_pearson_corrs, all_val_pearson_corrs

all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, all_train_r2s, all_val_r2s, all_train_pearson_corrs, all_val_pearson_corrs = run_k_fold_cross_validation(pyg_data_list, k=10, epochs=150, batch_size=32)

# Plotting
plt.figure(figsize=(20, 25))

# Plotting training and validation loss
plt.subplot(4, 2, 1)
for i, train_loss in enumerate(all_train_losses):
    plt.plot(train_loss, label=f'Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(4, 2, 2)
for i, val_loss in enumerate(all_val_losses):
    plt.plot(val_loss, label=f'Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(4, 2, 3)
for i, train_accuracy in enumerate(all_train_accuracies):
    plt.plot(train_accuracy, label=f'Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(4, 2, 4)
for i, val_accuracy in enumerate(all_val_accuracies):
    plt.plot(val_accuracy, label=f'Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

# Plotting R2 Score
plt.subplot(4, 2, 5)
for i, val_r2 in enumerate(all_val_r2s):
    plt.plot(val_r2, label=f'Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('R2 Score')
plt.title('Validation R2 Score')
plt.legend()

# Plotting Pearson Correlation
plt.subplot(4, 2, 6)
for i, val_pearson_corr in enumerate(all_val_pearson_corrs):
    plt.plot(val_pearson_corr, label=f'Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Pearson Correlation')
plt.title('Validation Pearson Correlation')
plt.legend()

plt.tight_layout()
plt.show()
