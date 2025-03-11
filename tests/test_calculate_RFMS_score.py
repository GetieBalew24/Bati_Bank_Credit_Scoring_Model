import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts.calculate_RFMS_score import RFMSRiskClassifier

class TestRFMSRiskClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample test data
        data = {
            'TransactionId': ['T1', 'T2', 'T3'],
            'CustomerId': ['C1', 'C1', 'C2'],
            'TransactionStartTime': ['2019-01-01', '2019-02-01', '2019-01-15'],
            'Transaction_Count': [2, 3, 1],
            'Total_Transaction_Amount': [100, 150, 200],
            'Transaction_Month': [1, 2, 1]
        }
        cls.df = pd.DataFrame(data)
        cls.classifier = RFMSRiskClassifier(cls.df)

    def test_calculate_recency(self):
        result = self.classifier.calculate_recency('2019-03-01')
        self.assertIn('Recency', result.columns)
        self.assertEqual(result['Recency'].max(), 59)

    def test_calculate_frequency(self):
        result = self.classifier.calculate_frequency()
        self.assertIn('Frequency', result.columns)
        self.assertEqual(result['Frequency'].iloc[0], 2)

    def test_calculate_monetary(self):
        result = self.classifier.calculate_monetary()
        self.assertIn('Monetary', result.columns)
        self.assertEqual(result[result['CustomerId'] == 'C1']['Monetary'].iloc[0], 250)

    def test_calculate_seasonality(self):
        result = self.classifier.calculate_seasonality()
        self.assertIn('Seasonality', result.columns)
        self.assertEqual(result[result['CustomerId'] == 'C1']['Seasonality'].iloc[0], 2)

    def test_normalize_rfms(self):
        self.classifier.calculate_recency('2019-03-01')
        self.classifier.calculate_frequency()
        self.classifier.calculate_monetary()
        self.classifier.calculate_seasonality()
        result = self.classifier.normalize_rfms()
        for col in ['Recency', 'Frequency', 'Monetary', 'Seasonality']:
            self.assertTrue(0 <= result[col].min() <= 1)
            self.assertTrue(0 <= result[col].max() <= 1)

    def test_assign_risk_category(self):
        self.classifier.calculate_recency('2019-03-01')
        self.classifier.calculate_frequency()
        self.classifier.calculate_monetary()
        self.classifier.calculate_seasonality()
        self.classifier.normalize_rfms()
        result = self.classifier.assign_risk_category(threshold=0.5)
        self.assertIn('Risk_category', result.columns)
        self.assertTrue(set(result['Risk_category']).issubset({'good', 'bad'}))

if __name__ == '__main__':
    unittest.main()