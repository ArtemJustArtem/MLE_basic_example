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
import time

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
    """
    Load the inference data

    Args:
        path: file path for inference features

    Returns: DataFrame of the inference features
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logging.exception("File with inference features was not found. Try to resolve this issue by running: "
                          "python3 data_process/data_generation.py")
        raise

def load_model(name):
    """
    Load the model

    Args:
        name: file path for model

    Returns: trained model
    """
    model = os.path.join(MODEL_DIR, name)
    if not os.path.exists(model):
        logging.error(f"Modelfile with the name {model} was not found. Check if the name was correct.")
        raise FileNotFoundError(f"Modelfile with the name {model} was not found.")
    with open(model, 'rb') as f:
        return pickle.load(f)
    
def get_predictions(model, dataset, device):
    """
    Infer predictions from the model

    model: trained model
    dataset: DataFrame of the inference features
    device: device used for calculations

    Returns: Series with predictions
    """
    X = torch.tensor(dataset.values, dtype=torch.float32).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = np.argmax(model(X).cpu().numpy(), axis=1)
    label_dict = conf['inference']['target_labels']
    labels = pd.Series(predictions).apply(lambda item: label_dict[item])
    return labels

def save_predictions(results, name):
    """
    Save predictions in the file

    Args:
        results: Series with predictions
        name: file path for the saved predictions

    Returns: None
    """
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
    logging.info(f"Data successfully loaded. Size of the dataset: {len(data)}")
    model = load_model(args.model)
    logging.info("Model successfully loaded")
    device = conf['general']['device']
    start_time = time.time()
    labels = get_predictions(model, data, device)
    end_time = time.time()
    logging.info(f"Model finished making predictions after {end_time - start_time} sec.")
    save_predictions(labels, args.model)
    logging.info("Script finished successfully.")