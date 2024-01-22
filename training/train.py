"""
This script prepares the data, runs the training, and saves the model.
"""
import numpy as np
import pandas as pd

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
                    default=100)
parser.add_argument("--verbose_interval", type=int,
                    help="Interval between each epoch log (where -1 means no epoch log at all)",
                    default=50)
args = parser.parse_args()

def data_pipeline(X_train_path, y_train_path, test_size, random_state):
    try:
        X = pd.read_csv(X_train_path)
    except FileNotFoundError:
        logging.exception("File with training features was not found. Try to resolve this issue by running: "
                          "python3 data_process/data_generation.py")
        raise
    try:
        y = pd.read_csv(y_train_path)['Target']
    except FileNotFoundError:
        logging.exception("File with training targets was not found. Try to resolve this issue by running: "
                          "python3 data_process/data_generation.py")
        raise
    y_encoded = pd.get_dummies(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state, stratify=y)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    return train_dataset, test_dataset

class NNModel(nn.Module):
    def __init__(self, hidden_neurons, device):
        super(NNModel, self).__init__()
        self.device = device
        neurons = hidden_neurons.split(',')
        self.hidden_layers = []
        in_features = 4
        for n in neurons:
            try:
                out_features = int(n)
            except ValueError:
                logging.exception(f"An error occured while trying to convert {n} to integer. Change the value of --hidden_layer")
                raise
            if out_features <= 0:
                logging.exception(f"Number of neurons must be positive integer but got {out_features}. Change the value of --hidden_layer")
                raise ValueError(f"Number of neurons must be positive integer but got {out_features}.")
            self.hidden_layers.append(nn.Linear(in_features=in_features, out_features=out_features, bias=True).to(self.device))
            in_features = out_features
        self.final_layer = nn.Linear(in_features=in_features, out_features=3, bias=True).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        return self.softmax(self.final_layer(x))

def evaluate_model(model, dataset):
    model.eval()
    with torch.no_grad():
        X, y = dataset[:][0], dataset[:][1]
        y = torch.tensor(np.argmax(y.cpu().numpy(), axis=1)).type(torch.int)
        predictions = torch.tensor(np.argmax(model(X).cpu().numpy(), axis=1)).type(torch.int)
        return multiclass_accuracy(predictions, y).item()
    
def train_model(model, train_dataset, batch_size, epochs, verbose_interval):
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for X, y in train_dataloader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.to(model.device))
            loss.backward()
            optimizer.step()
        if verbose_interval != -1 and epoch % verbose_interval == 0:
                logging.info(f"Epoch #{epoch + 1} -> loss: {loss.item()}")
    end_time = time.time()
    logging.info(f"Model finished training after {round(end_time - start_time, 2)} sec. Final loss: {loss.item()}")
    return model

def save_model(model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    name = datetime.now().strftime(conf['general']['datetime_format']) + '.pickle'
    path = os.path.join(MODEL_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
        logging.info(f"Model saved as '{name}'")

if __name__ == "__main__":
    configure_logging()
    logging.info("Starting the script.")
    np.random.seed(conf['general']['random_state'])
    torch.random.manual_seed(conf['general']['random_state'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf['general']['random_state'])
        torch.backends.cudnn.benchmark = False
    train_dataset, test_dataset = data_pipeline(X_TRAIN_PATH, Y_TRAIN_PATH, conf['train']['test_size'], conf['general']['random_state'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using {device} to train model")
    model = NNModel(args.hidden_neurons, device)
    logging.info("Starting training of the model. "
                 f"Parameters used: Number of neurons in hidden layers: [{args.hidden_neurons}]; "
                 f"Batch size: {args.batch_size}; "
                 f"Number of epochs: {args.epochs}")
    trained_model = train_model(model, train_dataset, args.batch_size, args.epochs, args.verbose_interval)
    logging.info(f"Train accuracy: {evaluate_model(trained_model, train_dataset)}")
    logging.info(f"Test accuracy: {evaluate_model(trained_model, test_dataset)}")
    save_model(trained_model)
    logging.info("Script successfully finished.")
    