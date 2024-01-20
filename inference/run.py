"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

from utils import get_project_dir, configure_logging
from training.train import NNModel

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])
INFERENCE_DATA = os.path.join(DATA_DIR, conf['inference']['inference_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str,
                    help="Filename of the model used to inference")
args = parser.parse_args()

def load_data(path):
    return pd.read_csv(path)

def load_model(name):
    model = os.path.join(MODEL_DIR, name)
    with open(model, 'rb') as f:
        return pickle.load(f)
    
def get_predictions(model, dataset, device):
    X = torch.tensor(dataset.values, dtype=torch.float32).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = np.argmax(model(X).cpu().numpy(), axis=1)
    label_dict = conf['inference']['target_labels']
    labels = pd.Series(predictions).apply(lambda item: label_dict[item])
    return labels

def save_predictions(results, name):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    file_path = os.path.join(RESULTS_DIR, name + '.csv')
    results = pd.DataFrame(results, columns=['Labels'])
    results.to_csv(file_path, index=False)
    logging.info(f"Predictions are saved to {name + '.csv'}")

if __name__ == "__main__":
    configure_logging()
    logging.info("Starting the script.")
    data = load_data(INFERENCE_DATA)
    logging.info("Data successfully loaded")
    model = load_model(args.model)
    logging.info("Model successfully loaded")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Used {device} to make predictions")
    labels = get_predictions(model, data, device)
    save_predictions(labels, args.model)
    logging.info("Script finished successfully.")