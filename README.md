Credit Card Fraud Detection

## Overview

This project, developed as part of my internship at **Eztec**, focuses on detecting fraudulent transactions in credit card data using machine learning techniques. The project demonstrates the entire machine learning pipeline—from data preprocessing and exploratory data analysis (EDA) to training and evaluating various classification models.

The goal of this project is to predict whether a given transaction is fraudulent or non-fraudulent, using a dataset of anonymized credit card transactions. This project involves using popular machine learning models such as **Logistic Regression**, **Random Forest Classifier**, and **Decision Tree Classifier** for fraud detection.

## Dataset

The dataset used in this project is sourced from **Kaggle**, and it contains anonymized credit card transactions labeled as either fraudulent (Class 1) or non-fraudulent (Class 0). You can access the dataset here: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets).

### Dataset Features
- **Time**: Time elapsed between the transaction and the first transaction in the dataset.
- **V1-V28**: Anonymized features derived from PCA (Principal Component Analysis) transformations of transaction data.
- **Amount**: The monetary value of the transaction.
- **Class**: The target variable, where **0** denotes a non-fraudulent transaction and **1** denotes a fraudulent transaction.

## Project Steps

### 1. **Libraries and Dependencies**
The project utilizes several key libraries for data manipulation, analysis, visualization, and machine learning:
- **Pandas**: For loading and manipulating data.
- **NumPy**: For numerical operations.
- **Seaborn**: For data visualization.
- **Scikit-learn (sklearn)**: For model building, training, and evaluation.

### 2. **Data Loading**
The dataset is loaded using **Pandas**, and for quicker testing, a subset of the data is used (1000 records in this example).

```python
import pandas as pd
import numpy as np

# Load the data (example with 1000 rows)
data = pd.read_csv("creditcard.csv", nrows=1000)
```

### 3. **Data Preprocessing**
- **Scaling & Standardization**: The features are scaled using **StandardScaler** to ensure that all features contribute equally to the model’s performance.
- **Handling Imbalanced Data**: Since the dataset is highly imbalanced (with more non-fraudulent transactions), techniques like **SMOTE** or **undersampling** can be explored to improve model performance.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

### 4. **Exploratory Data Analysis (EDA)**
Basic exploratory data analysis is performed to understand the distribution of fraudulent vs. non-fraudulent transactions. **Seaborn** is used to create visualizations.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example plot: Visualize the distribution of fraud vs non-fraud transactions
sns.countplot(x='Class', data=data)
plt.show()
```

### 5. **Model Training**
Three machine learning models are trained to predict fraudulent transactions:
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree Classifier**

Each model is trained using **Scikit-learn**, and the dataset is split into **training** and **testing** sets using **train_test_split**.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Example: Initialize models
log_model = LogisticRegression()
rf_model = RandomForestClassifier()
dt_model = DecisionTreeClassifier()
```

### 6. **Model Evaluation**
The models are evaluated using several performance metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

These metrics help assess how well the models are detecting fraudulent transactions. The metrics are calculated using **Scikit-learn's** `accuracy_score`, `f1_score`, `precision_score`, and `recall_score`.

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Example: Evaluate Logistic Regression model
log_model.fit(X_train, y_train)
log_predictions = log_model.predict(X_test)

accuracy = accuracy_score(y_test, log_predictions)
f1 = f1_score(y_test, log_predictions)
precision = precision_score(y_test, log_predictions)
recall = recall_score(y_test, log_predictions)
```

### 7. **Results**
A comparative analysis of the models is performed based on the evaluation metrics, helping identify the most effective model for fraud detection.

## Requirements

To run this project locally, make sure you have the following dependencies installed. You can install them using **pip**:

```bash
pip install pandas numpy seaborn scikit-learn
```

## Conclusion

This project showcases how machine learning techniques can be applied to detect fraudulent credit card transactions. It covers the entire pipeline, from data preprocessing to training models and evaluating their performance. With further tuning and cross-validation, the models can be refined for improved accuracy and reliability in real-world fraud detection systems.

---

### Additional Suggestions:
- **Hyperparameter Tuning**: You can further optimize the model’s performance using **GridSearchCV** or **RandomizedSearchCV**.
- **Cross-validation**: Implement cross-validation to ensure the model's performance is consistent across different datasets.
- **Model Deployment**: Consider deploying the model in a production environment, where it can evaluate transactions in real-time.
