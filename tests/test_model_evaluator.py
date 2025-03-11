import sys
import unittest
import pandas as pd
import numpy as np
import logging
import os
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import scorecardpy as sc

# Import the ModelEvaluator class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts.WoE_evaluate import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and environment for all tests."""
        # Create a synthetic dataset for testing
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        cls.df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        cls.df['CustomerId'] = range(100)
        cls.df['Target'] = y

        # Add numerical features required by the ModelEvaluator
        cls.df['Total_Transaction_Amount'] = np.random.uniform(100, 1000, 100)
        cls.df['Avg_Transaction_Amount'] = np.random.uniform(10, 100, 100)
        cls.df['Transaction_Count'] = np.random.randint(1, 50, 100)
        cls.df['Std_Transaction_Amount'] = np.random.uniform(1, 10, 100)

        # Initialize ModelEvaluator
        cls.model_evaluator = ModelEvaluator(cls.df, target='Target')

    def setUp(self):
        """Set up before each test."""
        # Reset breaks for each test
        self.model_evaluator.breaks = {}

    def test_woe_num(self):
        """Test the woe_num method."""
        breaks = self.model_evaluator.woe_num()
        self.assertIsInstance(breaks, dict)
        self.assertGreater(len(breaks), 0)
        for col, breaks_list in breaks.items():
            self.assertIsInstance(breaks_list, list)
            self.assertGreater(len(breaks_list), 0)

    def test_adjust_woe(self):
        """Test the adjust_woe method."""
        # First, calculate breaks
        self.model_evaluator.woe_num()
        try:
            self.model_evaluator.adjust_woe()
        except Exception as e:
            self.fail(f"adjust_woe raised an exception: {e}")

    def test_woeval(self):
        """Test the woeval method."""
        # First, calculate breaks
        self.model_evaluator.woe_num()
        train_woe = self.model_evaluator.woeval(self.df)
        self.assertIsInstance(train_woe, pd.DataFrame)
        self.assertGreater(train_woe.shape[1], 0)

    def test_calculate_iv(self):
        """Test the calculate_iv method."""
        # First, calculate breaks and apply WoE transformation
        self.model_evaluator.woe_num()
        train_woe = self.model_evaluator.woeval(self.df)
        df_merged, iv_results = self.model_evaluator.calculate_iv(train_woe, 'Target')
        self.assertIsInstance(df_merged, pd.DataFrame)
        self.assertIsInstance(iv_results, pd.DataFrame)
        self.assertGreater(iv_results.shape[0], 0)

    def test_woe_num_with_missing_data(self):
        """Test the woe_num method with missing data."""
        # Introduce NaN values in one of the columns
        self.df.loc[:10, 'Total_Transaction_Amount'] = np.nan
        breaks = self.model_evaluator.woe_num()
        self.assertIsInstance(breaks, dict)
        self.assertNotIn('Total_Transaction_Amount', breaks)  # Column with NaN should be skipped

    def test_woe_num_with_infinite_data(self):
        """Test the woe_num method with infinite data."""
        # Introduce infinite values in one of the columns
        self.df.loc[:10, 'Avg_Transaction_Amount'] = np.inf
        breaks = self.model_evaluator.woe_num()
        self.assertIsInstance(breaks, dict)
        self.assertNotIn('Avg_Transaction_Amount', breaks)  # Column with inf should be skipped

    def test_woeval_with_invalid_input(self):
        """Test the woeval method with invalid input."""
        with self.assertRaises(ValueError):
            self.model_evaluator.woeval("invalid_input")  # Non-DataFrame input

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove logs directory
        if os.path.exists("../log"):
            for file in os.listdir("../log"):
                os.remove(os.path.join("../log", file))
            os.rmdir("../log")

if __name__ == '__main__':
    unittest.main()