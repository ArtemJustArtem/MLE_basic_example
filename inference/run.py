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

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])
INFERENCE_DATA = os.path.join(DATA_DIR, conf['inference']['inference_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str,
                    help="Filename of the model used to inference")
args = parser.parse_args()

def load_data(path):
    return pd.read_csv(path)

if __name__ == "__main__":
    configure_logging()
    logging.info("Starting the script.")
    _ = load_data(INFERENCE_DATA)
    logging.info("Script finished successfully.")