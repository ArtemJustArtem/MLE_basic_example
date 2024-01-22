import unittest
import pandas as pd
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = "settings.json"

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

import data_process.data_generation as data

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


if __name__ == '__main__':
    unittest.main()