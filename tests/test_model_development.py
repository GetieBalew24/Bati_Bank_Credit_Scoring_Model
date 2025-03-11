import sys
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import mlflow
import logging
import os
import shutil

# Import the ModelEvaluator class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts.model_development_scripts import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and environment for all tests."""
        # Create a synthetic dataset for testing
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        cls.data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        cls.data['CustomerId'] = range(100)
        cls.data['Target'] = y

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        cls.experiment_name = "Test_Model_Evaluation"
        mlflow.set_experiment(cls.experiment_name)

        # Initialize ModelEvaluator
        cls.model_evaluator = ModelEvaluator(cls.data, target_column='Target', experiment_name=cls.experiment_name)

    def setUp(self):
        """Set up before each test."""
        # Split the data
        self.model_evaluator.split_data(test_size=0.2, random_state=42)

    def test_split_data(self):
        """Test the split_data method."""
        X_train, X_test, y_train, y_test = self.model_evaluator.split_data()
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

    def test_train_logistic_regression(self):
        """Test the train_logistic_regression method."""
        model = self.model_evaluator.train_logistic_regression()
        self.assertIsInstance(model, LogisticRegression)
        self.assertIn('Logistic Regression', self.model_evaluator.models)

    def test_train_random_forest(self):
        """Test the train_random_forest method."""
        model = self.model_evaluator.train_random_forest()
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertIn('Random Forest', self.model_evaluator.models)

    def test_save_model(self):
        """Test the save_model method."""
        model = LogisticRegression()
        model_name = "test_model"
        self.model_evaluator.save_model(model, model_name)
        self.assertTrue(os.path.exists(f"models/{model_name}.pkl"))
        # Clean up
        os.remove(f"models/{model_name}.pkl")

    def test_plot_roc_curves(self):
        """Test the plot_roc_curves method."""
        # Train models first
        self.model_evaluator.train_logistic_regression()
        self.model_evaluator.train_random_forest()
        # Test plotting
        try:
            self.model_evaluator.plot_roc_curves()
        except Exception as e:
            self.fail(f"plot_roc_curves raised an exception: {e}")

    def test_display_classification_reports(self):
        """Test the display_classification_reports method."""
        # Train models first
        self.model_evaluator.train_logistic_regression()
        self.model_evaluator.train_random_forest()
        # Test displaying reports
        try:
            self.model_evaluator.display_classification_reports()
        except Exception as e:
            self.fail(f"display_classification_reports raised an exception: {e}")

    def test_plot_model_comparisons(self):
        """Test the plot_model_comparisons method."""
        # Train models first
        self.model_evaluator.train_logistic_regression()
        self.model_evaluator.train_random_forest()
        # Test plotting
        try:
            self.model_evaluator.plot_model_comparisons()
        except Exception as e:
            self.fail(f"plot_model_comparisons raised an exception: {e}")

    def test_plot_metric_comparisons(self):
        """Test the plot_metric_comparisons method."""
        # Train models first
        self.model_evaluator.train_logistic_regression()
        self.model_evaluator.train_random_forest()
        # Test plotting
        try:
            self.model_evaluator.plot_metric_comparisons()
        except Exception as e:
            self.fail(f"plot_metric_comparisons raised an exception: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove MLflow artifacts
        if os.path.exists("/tmp/mlruns"):
            shutil.rmtree("/tmp/mlruns")
        # Remove logs directory
        if os.path.exists("../logs"):
            shutil.rmtree("../logs")
        # Remove models directory
        if os.path.exists("models"):
            shutil.rmtree("models")

if __name__ == '__main__':
    unittest.main()