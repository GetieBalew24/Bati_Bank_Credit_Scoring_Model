# Bati-Bank-Credit-Scoring-Model

## Overview
This repository contains a project for developing a **Credit Scoring Model** using the RFMS formalism. The project is focused on classifying customers into "high risk" and "low risk" groups based on **Recency (R)**, **Frequency (F)**, **Monetary (M)**, and **Subscription Status (S)** behaviors. This credit scoring model helps determine customer creditworthiness by analyzing transaction data from an eCommerce platform, enabling a **Buy Now, Pay Later** service.

The main objective is to classify users as **good** (low-risk) or **bad** (high-risk) based on their behavior and to assign a **credit score** to each customer accordingly. The model includes features like **Weight of Evidence (WoE) binning** and risk probability estimation, followed by credit score calculation.

## Project Structure
- **data/**: This folder contains the dataset used in your project.
- **notebooks/**: Jupyter notebooks that were used for exploratory data analysis (EDA) or model development.
- **scripts/**: This folder contains helper scripts that are used for various tasks like data preprocessing, model training, or EDA scripts.
- **src/**: Python modules or scripts for app.
- **tests/**: Contains test scripts for validating the functionalities of your code.
- **README.md**: The documentation file that explains the project.
- **requirements.txt**: A file that lists the Python packages needed to run the project.
### Data
The data used in this project contains transactional information, including:
- `TransactionId`: Unique transaction identifier
- `BatchId`: Batch of transactions
- `AccountId`: Customer account ID
- `SubscriptionId`: Subscription ID
- `CustomerId`: Customer ID
- `CurrencyCode`: Transaction currency
- `CountryCode`: Customer's country code
- `ProviderId`: Source provider for the product bought
- `ProductId`: Product being purchased
- `ProductCategory`: Broader category for the product
- `ChannelId`: Channel used (Web, Android, IOS, etc.)
- `Amount`: Transaction value (positive for debits, negative for credits)
- `Value`: Absolute transaction value
- `TransactionStartTime`: Start time of the transaction
- `PricingStrategy`: Pricing structure category
- `FraudResult`: Fraud detection result (1 - Fraud, 0 - No Fraud)

## Features
1. **Exploratory Data Analysis (EDA)**:
   - Understand the structure of the dataset, summary statistics, distribution of features, correlation analysis, missing value identification, and outlier detection.