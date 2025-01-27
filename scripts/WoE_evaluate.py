import numpy as np
import pandas as pd
import scorecardpy as sc
from monotonic_binning.monotonic_woe_binning import Binning  
import os
import logging

# Create a 'log' folder if it doesn't exist
if not os.path.exists('../log'):
    os.makedirs('../log')

# Set up logging configuration
logging.basicConfig(
    filename='../log/model_evaluator.log',  # Log file path
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
)

class ModelEvaluator:
    def __init__(self, df, target):
        """
        Initialize the ModelEvaluator class with a DataFrame and the target column.
        Args:
            df (pd.DataFrame): The dataset containing features and target.
            target (str): The target column name (binary classification).
        """
        self.df = df
        self.target = target
        self.breaks = {}
        logging.info('ModelEvaluator initialized with dataframe of shape %s and target column %s', df.shape, target)
