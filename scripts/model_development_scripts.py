import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
import logging
from datetime import datetime
import mlflow
import mlflow.sklearn


class ModelEvaluator:
    def __init__(self, data, target_column, experiment_name="Model_Evaluation"):
        self.data = data
        self.target_column = target_column
        self.models = {}
        self.results = {}
        self.classification_reports = {}

        # Setup logging
        self.setup_logging()
        logging.info("ModelEvaluator initialized.")
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        logging.info(f"MLflow experiment '{experiment_name}' initialized.")

    def setup_logging(self):
        """Set up logging to track operations and save logs to a file."""
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"model_evaluator_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )