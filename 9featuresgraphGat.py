import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Load the list of wild types, mutant types, and labels
data_path = "/home/msc2021_23/Data/Mahesh/sanjay/eg_graph.csv"
data = pd.read_csv(data_path)

def load_graph_data(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)

def create_edge_index(edges):
    edge_index = []
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        edge_index.append([source, target])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

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

    return torch.tensor(feature_matrix, dtype=torch.float)

def process_graph(file_path):
    graph_data = load_graph_data(file_path)
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    edge_index = create_edge_index(edges)
    feature_matrix = create_feature_matrix(nodes)
    return edge_index, feature_matrix

# Directories containing JSON files
wild_directory = "/home/msc2021_23/Data/Mahesh/sanjay/new_graphs/wild_graphs/"
mutant_directory = "/home/msc2021_23/Data/Mahesh/sanjay/new_graphs/"

graph_pairs = []

for index, row in data.iterrows():
    wild_file = os.path.join(wild_directory, row['wildtype_graph'])
    mutant_file = os.path.join(mutant_directory, row['Mutated graphs'])
    label = row['ddG_ML']
    
    wild_edge_index, wild_features = process_graph(wild_file)
    mutant_edge_index, mutant_features = process_graph(mutant_file)
    
    graph_pairs.append((wild_edge_index, wild_features, mutant_edge_index, mutant_features, label))

def to_pyg_data(edge_index, feature_matrix, label):
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=feature_matrix, edge_index=edge_index, y=y)

pyg_data_list = []

for wild_edge_index, wild_features, mutant_edge_index, mutant_features, label in graph_pairs:
    wild_data = to_pyg_data(wild_edge_index, wild_features, label)
    mutant_data = to_pyg_data(mutant_edge_index, mutant_features, label)
    pyg_data_list.append((wild_data, mutant_data))

# Create DataLoader
batch_size = 32
data_loader = DataLoader(pyg_data_list, batch_size=batch_size, shuffle=True)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

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
    return total_loss / len(data_loader), mse, r2, pearson_corr, all_targets, all_predictions

# Instantiate the model, optimizer, and loss function
in_channels = 9  # Number of node features
out_channels = 1  # Output size (regression to a single value)

model = GAT(in_channels, out_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression

# Training loop
epochs = 150

losses = []
mses = []
r2s = []
pearson_corrs = []

for epoch in range(epochs):
    loss, mse, r2, pearson_corr, _, _ = train(model, data_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}, Pearson Corr: {pearson_corr:.4f}")
    losses.append(loss)
    mses.append(mse)
    r2s.append(r2)
    pearson_corrs.append(pearson_corr)

# Plotting
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(mses, label='MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE)')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(pearson_corrs, label='Pearson Correlation')
plt.xlabel('Epoch')
plt.ylabel('Pearson Correlation')
plt.title('Pearson Correlation')
plt.legend()

plt.tight_layout()
plt.show()

