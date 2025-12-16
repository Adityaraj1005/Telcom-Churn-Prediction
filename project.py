import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Step 1: Load the dataset ---
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# --- Step 2: Separate features and target variable ---
X = dataset.iloc[:, 1:-1].values  # All columns except CustomerID and Churn
y = dataset.iloc[:, -1].values    # Target column: Churn

# --- Step 3: Split the dataset into training and test sets ---
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# --- Step 4: Encode categorical features using OneHotEncoder ---
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Indices of categorical columns in X
categorical_indices = [0,2,3,5,6,7,8,9,10,11,12,13,14,15,16]

# ColumnTransformer applies OneHotEncoder to categorical columns; remainder passed through
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(sparse_output=False), categorical_indices)],
    remainder='passthrough'
)

# Fit transformer on training set and transform both training and test sets
X_train = np.array(ct.fit_transform(X_train))
X_test  = np.array(ct.transform(X_test))

# --- Step 5: Encode target variable (Churn) to numeric values ---
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

# --- Step 6: Convert last 3 numeric columns to float and handle NaNs ---
# After one-hot encoding, the last 3 columns are numeric features (e.g., tenure, MonthlyCharges, TotalCharges)
X_train[:, -3:] = np.nan_to_num(
    pd.DataFrame(X_train[:, -3:]).apply(pd.to_numeric, errors='coerce'), nan=0.0
)
X_test[:, -3:]  = np.nan_to_num(
    pd.DataFrame(X_test[:, -3:]).apply(pd.to_numeric, errors='coerce'), nan=0.0
)

# --- Step 7: Feature scaling of numeric columns ---
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, -3:] = sc.fit_transform(X_train[:, -3:])  # Scale training numeric features
X_test[:, -3:]  = sc.transform(X_test[:, -3:])       # Scale test numeric features using same scaler

# --- Step 8: Train Logistic Regression model ---
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# --- Step 9: Predict on test set using Logistic Regression ---
y_pred = classifier.predict(X_test)

# --- Step 10: Evaluate Logistic Regression model ---
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Logistic Regression:")
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Step 11: Train Support Vector Machine (SVM) model ---
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# --- Step 12: Predict on test set using SVM ---
y_pred = classifier.predict(X_test)

# --- Step 13: Evaluate SVM model ---
cm = confusion_matrix(y_test, y_pred)
print("\nSVM (Linear Kernel):")
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
