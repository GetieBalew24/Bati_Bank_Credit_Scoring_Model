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
    def extract_temporal_features(self):
        """
        Extracts temporal features from the transaction timestamp.

        Returns:
            pd.DataFrame: DataFrame with new temporal feature columns.
        """
        self.df['Transaction_Hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['Transaction_Day'] = self.df['TransactionStartTime'].dt.day
        self.df['Transaction_Month'] = self.df['TransactionStartTime'].dt.month
        self.df['Transaction_Year'] = self.df['TransactionStartTime'].dt.year
        return self.df
    def handle_missing_values(self, method='imputation', strategy='mean', numerical_cols=None, categorical_cols=None):
        """
        Handles missing values in the DataFrame using specified methods.

        Args:
            method (str): The method to handle missing values ('imputation' or 'removal').
            strategy (str): The strategy for imputation ('mean', 'median', or 'mode').
            numerical_cols (list, optional): List of numerical columns to process.
            categorical_cols (list, optional): List of categorical columns to process.

        Returns:
            pd.DataFrame: DataFrame after handling missing values.
        """
        if numerical_cols is None:
            numerical_cols = self.df.select_dtypes(include=['float64', 'int']).columns.tolist()
        
        if categorical_cols is None:
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        # Handle numerical missing values
        if method == 'imputation':
            for col in numerical_cols:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
        elif method == 'removal':
            self.df.dropna(subset=numerical_cols, inplace=True)

        # Handle categorical missing values
        for col in categorical_cols:
            if method == 'imputation':
                self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)  # Filling with mode for categorical
            elif method == 'removal':
                self.df.dropna(subset=categorical_cols, inplace=True)

        return self.df