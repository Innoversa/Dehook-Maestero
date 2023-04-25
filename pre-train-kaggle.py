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
import wandb

import argparse


def str2bool(v: str) -> bool:
    """
    str2bool
        Convert String into Boolean

    Arguments:
        v {str} --input raw string of boolean command

    Returns:
        bool -- converted boolean value
    """
    return v.lower() in ("yes", "true", "y", "t", "1")


def sicong_argparse(model: str) -> argparse.Namespace:
    """
    sicong_argparse
        parsing command line arguments with reinforced formats

    Arguments:
        model {str} -- indicates which model being used

    Returns:
        argparse.Namespace -- flags containining the specification of this run
    """
    model_desc_dict = {
        "multi-modal": "Using multi-modal to predict macro nutrient",
        "cgm-only": "only using CGM readings to predict macro nutrient",
    }
    if model not in model_desc_dict:
        raise RuntimeError(
            "Model Unknown, only 'Sequnet' and 'Transformer' are available at this time"
        )
    parser = argparse.ArgumentParser(description=model_desc_dict[model])
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Choosing the Batch Size, default=32 (If out of memory, try a smaller batch_size)",
    )
    parser.add_argument(
        "--lstm_dim",
        type=int,
        default=32,
        help="dimension of LSTM model default=32",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=112,
        help="size of image input default=112",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight Decay hyperparameter default=0",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Choose the max number of epochs, default=3 for testing purpose",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Define Learning Rate, default=1e-4, if failed try something smaller",
    )
    parser.add_argument(
        "--ignore_first_meal",
        type=str2bool,
        default=False,
        help="Whether to ignore the first meal in the sequence (usually the first meal is a calibration meal))",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Define the dropout rate, default=0",
    )
    parser.add_argument("--save_result", type=str2bool, default=False)
    parser.add_argument(
        "--sel_gpu",
        type=int,
        default=4,
        help="Choosing which GPU to use (STMI has GPU 0~7)",
    )
    parser.add_argument(
        "--shuffle_data",
        type=str2bool,
        default=False,
        help="Whether to shuffle data before train/test split",
    )
    parser.add_argument(
        "--use_wandb",
        type=str2bool,
        default=False,
        help="Whether to save progress and results to Wandb",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="BSN-MacroNutrient",
        help="If using Wandb, this shows the location of the project, usually don't change this one",
    )
    parser.add_argument(
        "--wandb_tag",
        type=str,
        default="default_sequnet",
        help="If using Wandb, define tag to help filter results",
    )
    flags, _ = parser.parse_known_args()
    # setting cuda device
    flags.device = f"cuda:{flags.sel_gpu}" if flags.sel_gpu > 0 else "cpu"
    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))
    return flags


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


def train(model, inputs, labels, optimizer, criterion, epochs, batch_size, flags):
    # split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        inputs, labels, test_size=0.1, random_state=42
    )
    log_dict = {}
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

        if epoch % 10 == 0:
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
        wandb.log(
            {
                "Train BCE Loss": train_loss,
                "Val BCE Loss": val_loss,
                "Train F1 Score": train_f1,
                "Val F1 Score": val_f1,
                "Val AUC ROC": val_auc_roc,
                "Val AUC PR": val_auc_pr,
            }
        )


if __name__ == "__main__":
    flags = sicong_argparse("multi-modal")
    cuda_device = "cuda:3"
    wandb.init(
        project="Accessible Computing",
        reinit=True,
    )
    wandb.config.update(flags)
    log_dict = {}
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
        flags=flags,
    )
    torch.save(model.cpu(), "pre-trained model with kaggle.pt")
