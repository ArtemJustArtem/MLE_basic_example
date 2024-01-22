import unittest
import pandas as pd
import os
import sys
import json
import torch
from utils import get_project_dir

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = "settings.json"

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = get_project_dir(conf['general']['data_dir'])
X_TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['X_train_name'])
Y_TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['y_train_name'])

import data_process.data_generation as data
import training.train as train

DEFAULT_MODEL = train.NNModel("8,4,2", "cpu")
TRAIN_DATASET, TEST_DATASET = train.data_pipeline(X_TRAIN_PATH, Y_TRAIN_PATH, 
                                                  conf['train']['test_size'], conf['general']['random_state'])

class DataProcessTests(unittest.TestCase):
    def test_checking_dataloader_X_train(self):
        gen = data.DataGenerator()
        X_train_df, _, _ = gen.load_data(conf['inference']['inference_size'], conf['general']['random_state'])
        self.assertIsInstance(X_train_df, pd.DataFrame, "Training features data is not DataFrame")
        cols = X_train_df.columns.to_list()
        self.assertListEqual(cols, ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
                             "Columns in training features data are incorrect")

    def test_checking_dataloader_y_train(self):
        gen = data.DataGenerator()
        _, y_train_df, _ = gen.load_data(conf['inference']['inference_size'], conf['general']['random_state'])
        self.assertIsInstance(y_train_df, pd.DataFrame, "Training targets data is not DataFrame")
        cols = y_train_df.columns.to_list()
        self.assertListEqual(cols, ["Target"],
                             "Columns in training targets data are incorrect")

    def test_checking_dataloader_X_inference(self):
        gen = data.DataGenerator()
        _, _, X_inference_df = gen.load_data(conf['inference']['inference_size'], conf['general']['random_state'])
        self.assertIsInstance(X_inference_df, pd.DataFrame, "Inference features data is not DataFrame")
        cols = X_inference_df.columns.to_list()
        self.assertListEqual(cols, ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
                             "Columns in inference features data are incorrect")

class TrainingTest(unittest.TestCase):
    def test_checking_incorrect_hidden_layers_nonint(self):
        with self.assertRaises(ValueError, msg="Non integer value for number of neurons must raise ValueError"):
            _ = train.NNModel("1,l,1", "cpu")

    def test_checking_incorrect_hidden_layers_nonpositive(self):
        with self.assertRaises(ValueError, msg="Non positive value for number of neurons must raise ValueError"):
            _ = train.NNModel("16,0,4", "cpu")

    def test_checking_incorrect_batch_size_nonpositive(self):
        with self.assertRaises(ValueError, msg="Non positive value for batch size must raise ValueError"):
            _ = train.train_model(DEFAULT_MODEL, TRAIN_DATASET, -32, 100, 50)

    def test_checking_incorrect_batch_size_too_big(self):
        with self.assertRaises(ValueError, msg="Batch size bigger than dataset size must raise ValueError"):
            _ = train.train_model(DEFAULT_MODEL, TRAIN_DATASET, 10000, 100, 50)

    def test_checking_incorrect_epochs(self):
        with self.assertRaises(ValueError, msg="Non positive value for number of epochs must raise ValueError"):
            _ = train.train_model(DEFAULT_MODEL, TRAIN_DATASET, 32, -100, 50)

    def test_checking_incorrect_verbose_interval(self):
        with self.assertRaises(ValueError, msg="Non positive value for verbose interval must raise ValueError"):
            _ = train.train_model(DEFAULT_MODEL, TRAIN_DATASET, 32, 100, -50)

    def test_checking_trained_model(self):
        model = train.train_model(DEFAULT_MODEL, TRAIN_DATASET, 32, 100, -1)
        self.assertIsInstance(model, train.NNModel, "Trained model must be NNModel type")


if __name__ == '__main__':
    unittest.main()