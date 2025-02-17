"""
This script is responsible for loading and saving training and inference datasets
"""

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
    """
    Singleton class for data generation
    """
    def load_data(self, inference_size, random_state):
        """
        Loading the data from scikit-learn

        Args:
            inference_size: part of the dataset to save as inference
            random_state: random state (used for reproducibility)

        Returns:
            X_train_df: DataFrame of training features
            X_inference_df: DataFrame of inference features
            y_train_df: DataFrame of training targets
        """
        iris = load_iris()
        X = iris['data']
        y = iris['target']
        feature_names = iris['feature_names']
        X_train, X_inference, y_train, _ = train_test_split(X, y, test_size=inference_size, random_state=random_state, stratify=y)
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_inference_df = pd.DataFrame(y_train, columns=['Target'])
        y_train_df = pd.DataFrame(X_inference, columns=feature_names)
        return X_train_df, X_inference_df, y_train_df

    def save_data(self, X_train_path, y_train_path, inference_path, inference_size, random_state):
        """
        Saving the datasets into .csv files

        Args:
            X_train_path: file path for training features
            y_train_path: file path for training targets
            inference_path: file path for inference features
            inference_size: part of the dataset to save as inference
            random_state: random state (used for reproducibility)

        Returns: None
        """
        X_train_df, X_inference_df, y_train_df = self.load_data(inference_size, random_state)
        X_train_df.to_csv(X_train_path, index=False)
        X_inference_df.to_csv(y_train_path, index=False)
        y_train_df.to_csv(inference_path, index=False)
        logging.info(f"Training features loaded in {conf['train']['X_train_name']}")
        logging.info(f"Training targets loaded in {conf['train']['y_train_name']}")
        logging.info(f"Inference features loaded in {conf['inference']['inference_name']}")

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting loading data...")
    gen = DataGenerator()
    gen.save_data(X_TRAIN_PATH, Y_TRAIN_PATH, INFERENCE_PATH, conf['inference']['inference_size'], conf['general']['random_state'])
    logger.info("Data loaded successfully.")