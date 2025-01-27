import numpy as np
import pandas as pd
import os
import logging

# Create a 'log' folder if it doesn't exist
if not os.path.exists('../log'):
    os.makedirs('../log')

# Set up logging configuration
logging.basicConfig(
    filename='../log/rfms_classifier.log',  # Log file path
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
)

class RFMSRiskClassifier:
    def __init__(self, df):
        """
        Initialize the RFMSRiskClassifier with a DataFrame.
        
        Parameters:
        - df: DataFrame containing transaction data with customer information.
        """
        self.df = df
        logging.info('RFMSRiskClassifier initialized with dataframe of shape %s', df.shape)

    def calculate_recency(self, current_date):
        """
        Calculate the recency (days since last transaction) for each customer.

        Parameters:
        - current_date: The current date used to calculate recency.

        Returns:
        - DataFrame with a new column 'Recency' representing the recency for each customer.
        """
        logging.info('Calculating recency...')
        try:
            self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'], errors='coerce').dt.tz_localize(None)
            current_date = pd.to_datetime(current_date).tz_localize(None)
            self.df['Recency'] = (current_date - self.df['TransactionStartTime']).dt.days
            logging.info('Recency calculation successful, resulting dataframe shape: %s', self.df.shape)
        except Exception as e:
            logging.error('Error calculating recency: %s', e)
        
        return self.df
    def calculate_frequency(self):
        """
        Calculate the frequency (transaction count) for each customer.

        Returns:
        - DataFrame with a new column 'Frequency' representing the transaction count for each customer.
        """
        logging.info('Calculating frequency...')
        try:
            self.df['Frequency'] = self.df['Transaction_Count']
            logging.info('Frequency calculation successful, resulting dataframe shape: %s', self.df.shape)
        except Exception as e:
            logging.error('Error calculating frequency: %s', e)
        
        return self.df
    def calculate_monetary(self):
        """
        Calculate the monetary value (total transaction amount) for each customer.

        Returns:
        - DataFrame with a new column 'Monetary' representing the total transaction amount for each customer.
        """
        logging.info('Calculating monetary value...')
        try:
            self.df['Monetary'] = self.df.groupby('CustomerId')['Total_Transaction_Amount'].transform('sum')
            logging.info('Monetary calculation successful, resulting dataframe shape: %s', self.df.shape)
        except Exception as e:
            logging.error('Error calculating monetary value: %s', e)
        
        return self.df
    def calculate_seasonality(self):
        """
        Calculate the seasonality (number of unique transaction months) for each customer.

        Returns:
        - DataFrame with a new column 'Seasonality' representing the unique months of transactions for each customer.
        """
        logging.info('Calculating seasonality...')
        try:
            self.df['Transaction_Month'].fillna(0, inplace=True)
            self.df['Seasonality'] = self.df.groupby('CustomerId')['Transaction_Month'].transform(lambda x: x.nunique())
            logging.info('Seasonality calculation successful, resulting dataframe shape: %s', self.df.shape)
        except Exception as e:
            logging.error('Error calculating seasonality: %s', e)
        
        return self.df