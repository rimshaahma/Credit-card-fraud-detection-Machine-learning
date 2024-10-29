
# Credit Card Fraud Detection

This project is part of my internship at Eztec, focusing on detecting fraudulent transactions in credit card data using machine learning techniques. The project demonstrates data preprocessing, exploratory data analysis, and various classification models to predict fraudulent transactions.

## Dataset
The dataset for this project is obtained from Kaggle. It contains anonymized credit card transactions labeled as either fraudulent or non-fraudulent. You can access the dataset here: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Project Steps

### 1. Libraries and Dependencies
The project utilizes the following libraries:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Seaborn**: For visualization.
- **Scikit-learn (sklearn)**: For implementing machine learning models, including regression and classifiers.

### 2. Data Loading
The data is loaded using `pandas.read_csv()` and limited to a smaller subset of records for quicker testing and development.

```python
import pandas as pd
import numpy as np

# Load the data (example with 1000 rows)
data = pd.read_csv("creditcard.csv", nrows=1000)
```

### 3. Data Preprocessing
Standardization and scaling of features are done using Scikit-learn's `StandardScaler` to ensure all features contribute equally to the model’s performance.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

### 4. Exploratory Data Analysis (EDA)
Basic EDA is performed to understand the distribution of fraudulent vs. non-fraudulent transactions. Seaborn is used for visualization.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example plot
sns.countplot(x='Class', data=data)
plt.show()
```

### 5. Model Training
Three models are trained to predict fraudulent transactions:
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree Classifier**

Each model is trained using `sklearn` and evaluated for performance.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
```

### 6. Model Evaluation
The models are evaluated based on the following metrics:
- **Accuracy**
- **F1 Score**
- **Precision**
- **Recall**

These metrics are calculated using Scikit-learn's `accuracy_score`, `f1_score`, `precision_score`, and `recall_score`.

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Example: Evaluate Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
```

## Results
The project provides a comparative analysis of the models, identifying the most effective model based on the evaluation metrics.

## Requirements
To install the required libraries, use the following command:

```bash
pip install pandas numpy seaborn scikit-learn
```

## Conclusion
This project demonstrates various machine learning techniques to detect fraudulent credit card transactions. With further tuning and cross-validation, the models can be refined for improved accuracy and reliability in detecting fraud.

---

**Note**: Please ensure you have downloaded the dataset from Kaggle and saved it as `creditcard.csv` in your project directory.
```

### Additional Suggestions:
- Save this `README.md` in the project’s root directory.
- Include a **Results** section once you have performance metrics to compare the models.

Let me know if you'd like further customization or help!
