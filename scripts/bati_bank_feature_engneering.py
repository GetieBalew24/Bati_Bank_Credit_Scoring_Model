import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class FeatureEngineering:
    def __init__(self, df):
        """
        Initializes the FeatureEngineering class with a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to be processed.
        """
        self.df = df

    def remove_high_missing_values(self, threshold=0.5):
        """
        Removes columns with missing values exceeding the specified threshold.

        Args:
            threshold (float): The percentage threshold for missing values (default is 0.5).

        Returns:
            pd.DataFrame: DataFrame after dropping columns with high missing values.
        """
        missing_percentage = self.df.isnull().mean()
        if (missing_percentage > threshold).any():
            print("Columns with missing values more than threshold:")
            print(missing_percentage[missing_percentage > threshold])
            self.df.drop(columns=missing_percentage[missing_percentage > threshold].index, inplace=True)
        else:
            print("No columns found with missing values more than threshold.")
        return self.df
    def create_aggregate_features(self):
        """
        Creates aggregate features based on transaction data.

        Returns:
            pd.DataFrame: DataFrame with new aggregate feature columns.
        """
        self.df['Total_Transaction_Amount'] = self.df.groupby('CustomerId')['Amount'].transform('sum')
        self.df['Avg_Transaction_Amount'] = self.df.groupby('CustomerId')['Amount'].transform('mean')
        self.df['Transaction_Count'] = self.df.groupby('CustomerId')['TransactionId'].transform('count')
        self.df['Std_Transaction_Amount'] = self.df.groupby('CustomerId')['Amount'].transform('std')
        return self.df