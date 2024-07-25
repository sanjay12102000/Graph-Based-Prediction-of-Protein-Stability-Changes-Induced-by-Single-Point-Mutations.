import os
import torch
import pickle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        return None

def run_k_fold_cross_validation(pyg_data_list, k=5, epochs=150, batch_size=32):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint("checkpoint.pth.tar")
    start_fold = 0
    start_epoch = 0
    if checkpoint is not None:
        start_fold = checkpoint['fold']
        start_epoch = checkpoint['epoch']
        all_train_losses = checkpoint['all_train_losses']
        all_val_losses = checkpoint['all_val_losses']
        all_train_accuracies = checkpoint['all_train_accuracies']
        all_val_accuracies = checkpoint['all_val_accuracies']
        all_train_r2s = checkpoint['all_train_r2s']
        all_val_r2s = checkpoint['all_val_r2s']
        all_train_pearson_corrs = checkpoint['all_train_pearson_corrs']
        all_val_pearson_corrs = checkpoint['all_val_pearson_corrs']
    else:
        all_train_losses = []
        all_val_losses = []
        all_train_accuracies = []
        all_val_accuracies = []
        all_train_r2s = []
        all_val_r2s = []
        all_train_pearson_corrs = []
        all_val_pearson_corrs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(pyg_data_list)):
        if fold < start_fold:
            continue  # Skip folds that are already completed
        print(f"Fold {fold + 1}/{k}")

        train_data = [pyg_data_list[i] for i in train_idx]
        val_data = [pyg_data_list[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        model = GATv2(in_channels, out_channels)
        if checkpoint is not None and fold == start_fold:
            model.load_state_dict(checkpoint['state_dict'])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression

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

            # Save checkpoint
            save_checkpoint({
                'fold': fold,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'all_train_losses': all_train_losses + [train_losses],
                'all_val_losses': all_val_losses + [val_losses],
                'all_train_accuracies': all_train_accuracies + [train_accuracies],
                'all_val_accuracies': all_val_accuracies + [val_accuracies],
                'all_train_r2s': all_train_r2s + [train_r2s],
                'all_val_r2s': all_val_r2s + [val_r2s],
                'all_train_pearson_corrs': all_train_pearson_corrs + [train_pearson_corrs],
                'all_val_pearson_corrs': all_val_pearson_corrs + [val_pearson_corrs]
            })

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accuracies.append(train_accuracies)
        all_val_accuracies.append(val_accuracies)
        all_train_r2s.append(train_r2s)
        all_val_r2s.append(val_r2s)
        all_train_pearson_corrs.append(train_pearson_corrs)
        all_val_pearson_corrs.append(val_pearson_corrs)

        # Save the model for the current fold
        torch.save(model.state_dict(), f"gatv2_fold_{fold+1}.pt")

        # Reset start_epoch for next fold
        start_epoch = 0

    return all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, all_train_r2s, all_val_r2s, all_train_pearson_corrs, all_val_pearson_corrs

# Plotting
all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, all_train_r2s, all_val_r2s, all_train_pearson_corrs, all_val_pearson_corrs = run_k_fold_cross_validation(pyg_data_list, k=10, epochs=150, batch_size=32)

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
