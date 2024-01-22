import unittest
import pandas as pd
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = os.getenv('CONF_PATH')

import data_process.data_generation as data


class DataProcessTests(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()