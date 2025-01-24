import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Ensure the log directory exists
log_dir = "../log"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "eda_analysis.log"), mode='w')
    ]
)

class ExploratoryDataAnalysis:
    def __init__(self, df):
        """
        Initializes the ExploratoryDataAnalysis class with a DataFrame.
        """
        self.df = df
        logging.info("ExploratoryDataAnalysis initialized with DataFrame of shape %s", df.shape)

    def head(self):
        """
        Returns the first five rows of the DataFrame.
        """
        logging.info("Fetching the first five rows of the DataFrame.")
        return self.df.head()
    def dataset_overview(self):
        """
        Displays an overview of the dataset, including the number of rows,
        number of columns, data types, and the first five rows.
        """
        logging.info("Displaying dataset overview.")
        print("Dataset Overview:")
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")
        print("\nColumn Data Types:")
        print(self.df.dtypes)
        print("\nFirst 5 rows of the dataset:")
        print(self.df.head())
    def summary_statistics(self):
        """
        Displays summary statistics for numerical columns in the dataset.
        """
        logging.info("Calculating summary statistics.")
        print("\nSummary Statistics:")
        print(self.df.describe())