"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torcheval.metrics.functional import multiclass_accuracy

# Comment this lines if you have problems with MLFlow installation
import mlflow
mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json" 

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
X_TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['X_train_name'])
Y_TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['y_train_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_neurons", type=str,
                    help="Number of neurons in each hidden layer (integers separated by comma)", 
                    default="8,4")
parser.add_argument("--batch_size", type=int,
                    help="Batch size used during training",
                    default=32)
parser.add_argument("--epochs", type=int,
                    help="Number of epochs used during training",
                    default=25)

def data_pipeline(X_train_path, y_train_path, test_size, random_state):
    X = pd.read_csv(X_train_path)
    y = pd.read_csv(y_train_path)['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    return train_dataset, test_dataset

class NNModel(nn.Module):
    def __init__(self, hidden_neurons):
        super(NNModel, self).__init__()
        neurons = hidden_neurons.split(',')
        self.hidden_layers = []
        in_features = 4
        for n in neurons:
            out_features = int(n)
            self.hidden_layers.append(nn.Linear(in_features=in_features, out_features=out_features, bias=True))
            in_features = out_features
        self.final_layer = nn.Linear(in_features=in_features, out_features=3, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        return self.softmax(self.final_layer(x))


if __name__ == "__main__":
    configure_logging()
    _ = NNModel("8,4,2")