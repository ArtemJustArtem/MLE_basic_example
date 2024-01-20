# Importing required libraries
import pandas as pd
import logging
import os
import sys
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
X_TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['X_train_name'])
Y_TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['y_train_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inference_name'])

# Singleton class for generating XOR data set
@singleton
class DataGenerator():
    def __init__(self):
        self.df = None

    def load_data(self, X_train_path, y_train_path, inference_path, inference_size, random_state):
        iris = load_iris()
        X = iris['data']
        y = iris['target']
        feature_names = iris['feature_names']
        X_train, X_inference, y_train, _ = train_test_split(X, y, test_size=inference_size, random_state=random_state, stratify=y)
        pd.DataFrame(X_train, columns=feature_names).to_csv(X_train_path, index=False)
        pd.DataFrame(y_train, columns=['Target']).to_csv(y_train_path, index=False)
        pd.DataFrame(X_inference, columns=feature_names).to_csv(inference_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting loading data...")
    gen = DataGenerator()
    gen.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, INFERENCE_PATH, conf['inference']['inference_size'], conf['general']['random_state'])
    logger.info("Data loaded successfully.")