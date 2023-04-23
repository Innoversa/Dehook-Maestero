import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def data_loader():
    kaggle_pt = torch.load("kaggle_hidden_states.pt").float()
    labels_pt = torch.from_numpy(np.array(torch.load("kaggle_labels.pt"))).float()
    print(f"kaggle_pt.shape: {kaggle_pt.shape}, labels_pt.shape: {labels_pt.shape}")
    return kaggle_pt, labels_pt


class MLP(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=64, out_channels=32, kernel_size=3, dilation=2
        )
        self.fc1 = nn.Linear(32 * 382, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)  # Add max pooling layer
        x = x.view(-1, 32 * 382)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc3(x)
        output = self.sigmoid(output)
        return output.squeeze(dim=1)


def train(model, inputs, labels, optimizer, criterion, epochs, batch_size):
    # split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        inputs, labels, test_size=0.1, random_state=42
    )

    num_train_samples = X_train.shape[0]
    num_val_samples = X_val.shape[0]

    num_train_batches = num_train_samples // batch_size
    num_val_batches = num_val_samples // batch_size

    for epoch in tqdm(range(epochs), desc="Epochs", ascii=True):
        train_loss = 0.0
        val_loss = 0.0

        # loop over training batches
        for i in range(num_train_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # get the current batch
            batch_inputs = X_train[start_idx:end_idx].to(cuda_device)
            batch_labels = y_train[start_idx:end_idx].to(cuda_device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward(retain_graph=True)  # add retain_graph=True here
            optimizer.step()
            with torch.no_grad():
                train_loss += loss.item()

        # loop over validation batches
        for i in range(num_val_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # get the current batch
            batch_inputs = X_val[start_idx:end_idx].to(cuda_device)
            batch_labels = y_val[start_idx:end_idx].to(cuda_device)

            # evaluate the model on validation set
            with torch.no_grad():
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        train_loss /= num_train_batches
        val_loss /= num_val_batches

        if epoch % 1 == 0:
            # get the predictions on the training set and calculate f1 score
            with torch.no_grad():
                train_preds = model(X_train.to(cuda_device)).cpu().numpy()
                train_f1 = f1_score(
                    y_train.cpu().numpy(), (train_preds >= 0.5).astype(int)
                )

            # get the predictions on the validation set and calculate f1 score, auc roc, and auc pr
            with torch.no_grad():
                val_preds = model(X_val.to(cuda_device)).cpu().numpy()
                val_f1 = f1_score(y_val.cpu().numpy(), (val_preds >= 0.5).astype(int))
                val_auc_roc = roc_auc_score(y_val.cpu().numpy(), val_preds)
                val_auc_pr = average_precision_score(y_val.cpu().numpy(), val_preds)

            tqdm.write(
                f"Epoch: {epoch+1}, Train BCE Loss: {train_loss:.2f}, "
                + f"Val BCE Loss: {val_loss:.2f}, F1 Score: {train_f1:.2f}/{val_f1:.2f}, "
                + f"AUC ROC: {val_auc_roc:.2f}, AUC PR: {val_auc_pr:.2f}"
            )


if __name__ == "__main__":
    cuda_device = "cuda:3"
    # load the data
    kaggle_pt, labels_pt = data_loader()
    # define your model
    model = MLP().to(cuda_device)
    # define your loss function
    criterion = nn.BCELoss()
    # define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # train your model for 100 epochs
    train(
        model,
        kaggle_pt,
        labels_pt,
        optimizer,
        criterion,
        epochs=100,
        batch_size=1024,
    )
    torch.save(model.cpu(), "pre-trained model with kaggle.pt")
