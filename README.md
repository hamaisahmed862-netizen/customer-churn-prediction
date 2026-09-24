# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn based on their demographic information, service usage, contract details, billing information, and customer lifetime value (CLTV).

The project uses a **neural network built with TensorFlow/Keras**, with preprocessing handled through **scikit-learn pipelines**. A **Streamlit web application** provides an interactive interface for making predictions for new customers.

## Live Demo

🚀 **Try the application:**
https://customer-churn-prediction-cv.streamlit.app/

---

## Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who are likely to leave can help businesses understand customer behavior and take appropriate retention actions.

This project builds an end-to-end churn prediction system:

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Train/Test Split
        ↓
Preprocessing Pipeline
        ↓
Neural Network
        ↓
Churn Probability
        ↓
Churn / No Churn
        ↓
Streamlit Application
```

---

## Dataset

The project uses the **IBM Telco Customer Churn dataset** available on Kaggle.

**Dataset:** Telco Customer Churn

The original dataset contains:

* **7,043 customers**
* **33 columns**
* Customer demographic information
* Services subscribed to
* Contract information
* Billing information
* Customer lifetime value
* Churn information

### Target Variable

The model predicts:

```text
Churn Value
```

where:

* `0` = No Churn
* `1` = Churn

The dataset contains:

| Class    | Customers | Percentage |
| -------- | --------: | ---------: |
| No Churn |     5,174 |     73.46% |
| Churn    |     1,869 |     26.54% |

Because the classes are not evenly distributed, accuracy alone is not sufficient for evaluating the model. Precision, recall, F1-score, and ROC-AUC are also considered.

---

## Features Used

The model uses 20 customer features.

### Customer Information

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure Months

### Services

* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### Contract & Billing

* Contract
* Paperless Billing
* Payment Method

### Financial Information

* Monthly Charges
* Total Charges
* CLTV

---

## Features Excluded

Some columns were excluded because they were not appropriate for the prediction model.

### CustomerID

An identifier rather than a meaningful predictive feature.

### Churn Label

This contains the same target information in text form and would duplicate the target.

### Churn Reason

This describes why a customer already churned and would introduce information that would not be available when predicting future churn.

### Churn Score

This is already an existing churn-risk score. Using it would make the model heavily dependent on another churn prediction system rather than learning the prediction from the customer features.

### Count

A constant column that does not provide useful information.

### Country and State

These columns contain no useful variation in this dataset.

### Geographic Features

The initial model excludes:

* City
* Zip Code
* Lat Long
* Latitude
* Longitude

This keeps the model focused on customer behavior, services, contracts, and billing characteristics.

---

## Exploratory Data Analysis

Exploratory data analysis was performed before model training to understand the dataset and identify data quality issues.

The analysis included:

* Dataset structure and dimensions
* Churn distribution
* Missing values
* Unique values
* Total Charges data type
* Churn by contract type
* Churn by internet service
* Churn by technical support
* Churn by customer tenure
* Monthly charges distribution

### Important Findings

Customers with **month-to-month contracts** have a substantially higher observed churn rate than customers with longer-term contracts.

Customers using **fiber optic internet** also show a higher observed churn rate than the other internet-service groups in this dataset.

Customers who churn have a higher average monthly charge than customers who do not churn.

---

## Data Cleaning

`Total Charges` was initially stored as an object/string column because some records contained blank values.

It was converted to numeric values:

```python
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)
```

This resulted in 11 missing values.

All 11 records correspond to customers with:

```text
Tenure Months = 0
```

The missing values are handled automatically by the preprocessing pipeline.

---

## Train/Test Split

The dataset was divided into:

* **80% training data**
* **20% testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The `stratify=y` parameter preserves approximately the same churn/non-churn distribution in both datasets.

The test set contains **1,409 customers**.

---

## Preprocessing

The project uses a scikit-learn `ColumnTransformer` containing separate pipelines for numerical and categorical features.

### Numerical Features

Numerical features are processed using:

1. Median imputation
2. StandardScaler

```text
Numerical Data
      ↓
Median Imputation
      ↓
StandardScaler
```

The numerical features are:

* Tenure Months
* Monthly Charges
* Total Charges
* CLTV

### Categorical Features

Categorical features are processed using:

1. Most-frequent-value imputation
2. One-hot encoding

```text
Categorical Data
       ↓
Most Frequent Imputation
       ↓
One-Hot Encoding
```

The encoder uses:

```python
handle_unknown="ignore"
```

This allows the preprocessing pipeline to handle unseen categorical values when making predictions on new customers.

After preprocessing, the 20 original features become **47 numerical inputs** for the neural network.

---

## Neural Network

The final model is a binary classification neural network built using TensorFlow/Keras.

### Architecture

```text
47 Input Features
       ↓
Dense Layer (32 neurons)
       ↓
ReLU
       ↓
Dense Layer (16 neurons)
       ↓
ReLU
       ↓
Dense Layer (1 neuron)
       ↓
Sigmoid
       ↓
Churn Probability
```

### Model Configuration

```text
Optimizer: Adam
Learning Rate: 0.001
Loss Function: Binary Crossentropy
Batch Size: 16
Maximum Epochs: 50
Early Stopping: Enabled
Class Weights: Balanced
```

### Activation Functions

The hidden layers use **ReLU** activation:

```text
ReLU(x) = max(0, x)
```

The output layer uses **Sigmoid**, which produces a probability between 0 and 1.

A probability of at least `0.5` is classified as:

```text
Churn
```

while a probability below `0.5` is classified as:

```text
No Churn
```

---

## Handling Class Imbalance

The dataset contains considerably more non-churn customers than churn customers.

To give the minority class more importance during training, balanced class weights were used:

```python
compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
```

This helps the neural network pay more attention to churn cases instead of primarily optimizing for the majority class.

---

## Early Stopping

Early stopping was used to reduce unnecessary training and help prevent overfitting.

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

The model monitors validation loss during training and restores the weights from the best validation-loss epoch.

---

# Model Evaluation

The final saved model was evaluated on the test set.

### Final Results

| Metric          |     Result |
| --------------- | ---------: |
| Test Accuracy   | **76.93%** |
| Churn Precision |  **55.1%** |
| Churn Recall    |  **70.3%** |
| Churn F1-Score  |  **61.8%** |
| ROC-AUC         | **84.98%** |

### Classification Report

| Class    | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| No Churn |      0.88 |   0.79 |     0.83 |
| Churn    |      0.55 |   0.70 |     0.62 |

---

## Confusion Matrix

The final confusion matrix was:

```text
                 Predicted
               No Churn  Churn

Actual
No Churn          821      214
Churn             111      263
```

Therefore:

* **821** non-churn customers were correctly classified.
* **214** non-churn customers were incorrectly classified as churn.
* **111** churn customers were incorrectly classified as non-churn.
* **263** churn customers were correctly identified.

---

## Threshold Analysis

The model produces a churn probability, allowing different classification thresholds to be examined.

| Threshold | Precision | Recall |    F1 |
| --------: | --------: | -----: | ----: |
|       0.3 |     0.486 |  0.856 | 0.620 |
|       0.4 |     0.524 |  0.778 | 0.626 |
|       0.5 |     0.551 |  0.703 | 0.618 |
|       0.6 |     0.607 |  0.623 | 0.615 |
|       0.7 |     0.654 |  0.516 | 0.577 |

The deployed application currently uses a threshold of **0.5**.

Changing the threshold changes the trade-off between precision and recall without retraining the neural network.

---

# Streamlit Application

The project includes an interactive Streamlit application.

The application allows a user to enter customer information such as:

* Customer demographics
* Tenure
* Internet service
* Technical support
* Contract type
* Payment method
* Monthly charges
* Total charges
* CLTV

After clicking **Predict Churn**, the application displays:

* Churn probability
* Churn / No Churn prediction

### Live Application

🚀 **https://customer-churn-prediction-cv.streamlit.app/**

The application loads the trained model and fitted preprocessing pipeline:

```text
Customer Input
      ↓
Saved Preprocessor
      ↓
Processed Features
      ↓
Saved Neural Network
      ↓
Churn Probability
      ↓
Prediction
```

---

# Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── telco_churn.xlsx
│
├── src/
│   └── preprocessing.py
│
├── eda.py
├── train.py
├── predict.py
├── app.py
│
├── model.keras
├── preprocessor.joblib
├── requirements.txt
└── README.md
```

### File Descriptions

| File                    | Purpose                       |
| ----------------------- | ----------------------------- |
| `data/telco_churn.xlsx` | Original dataset              |
| `src/preprocessing.py`  | Preprocessing pipelines       |
| `eda.py`                | Exploratory data analysis     |
| `train.py`              | Model training and evaluation |
| `predict.py`            | Standalone prediction example |
| `app.py`                | Streamlit web application     |
| `model.keras`           | Trained neural network        |
| `preprocessor.joblib`   | Fitted preprocessing pipeline |
| `requirements.txt`      | Python dependencies           |
| `README.md`             | Project documentation         |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/hamaisahmed862-netizen/customer-churn-prediction.git
```

Navigate into the project:

```bash
cd customer-churn-prediction
```

Create a virtual environment:

```bash
py -3.12 -m venv .venv
```

Activate it on Windows:

```bash
.\.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Exploratory Data Analysis

```bash
python eda.py
```

## Run Standalone Prediction

```bash
python predict.py
```

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TensorFlow**
* **Keras**
* **Matplotlib**
* **Seaborn**
* **OpenPyXL**
* **Joblib**
* **Streamlit**

---

# Machine Learning Concepts Used

This project demonstrates practical use of:

* Exploratory Data Analysis
* Data cleaning
* Missing-value handling
* Feature preprocessing
* One-hot encoding
* Feature scaling
* Train/test splitting
* Stratified sampling
* Scikit-learn `Pipeline`
* `ColumnTransformer`
* Neural networks
* ReLU activation
* Sigmoid activation
* Binary crossentropy
* Adam optimizer
* Backpropagation
* Class weights
* Early stopping
* Probability-based classification
* Classification threshold analysis
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Model persistence
* Streamlit deployment

---

# Key Learning Outcomes

Through this project, I gained practical experience in building an end-to-end machine learning workflow rather than only training a model.

The project covers the complete process from:

```text
Raw Dataset
     ↓
EDA
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Preprocessing
     ↓
Neural Network Training
     ↓
Evaluation
     ↓
Model Saving
     ↓
Prediction
     ↓
Streamlit Deployment
```

The project also demonstrates how a fitted preprocessing pipeline can be saved alongside a trained neural network so that the same transformations are applied when making predictions on new customer data.

---

# Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Additional feature engineering
* Testing additional neural-network architectures
* Calibration of predicted probabilities
* Further analysis of customer segments
* Model monitoring
* Prediction logging
* Further deployment improvements
