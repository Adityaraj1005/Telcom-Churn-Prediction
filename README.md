# Telco Customer Churn Prediction

I chose this project to understand how machine learning can be applied to solve real-world business problems. Customer churn is a critical issue in the telecom industry, and predicting which customers are likely to leave helps companies take preventive actions. Through this project, I wanted to practice end-to-end machine learning concepts such as data preprocessing, feature encoding, model training, and evaluation on a realistic dataset.

## Dataset

The dataset used in this project is the **Telco Customer Churn Dataset**, which contains information about customers of a telecom company. The dataset includes customer demographics, service subscriptions, account information, and billing details.

**Source:** IBM / Kaggle  

**Key features include:**
- Gender  
- SeniorCitizen  
- Tenure  
- Contract type  
- Payment method  
- MonthlyCharges  
- TotalCharges  

**Target variable:**
- `Churn` — indicates whether a customer left the service (`Yes` or `No`).



## Project Workflow

1. **Data Loading and Exploration**  
   Loaded the Telco Customer Churn dataset and examined the structure of the data, including categorical and numerical features.

2. **Feature Selection**  
   Removed the customer ID column as it does not contribute to churn prediction and separated input features (X) from the target variable (y).

3. **Target Variable Encoding**  
   Converted the target variable (Churn) from categorical labels (`Yes`, `No`) into numerical format using Label Encoding.

4. **Train-Test Split**  
   Split the dataset into training and testing sets to evaluate model performance on unseen data.

5. **Categorical Feature Encoding**  
   Applied One-Hot Encoding to categorical features so that machine learning models can process non-numeric data.

6. **Handling Numerical Features**  
   Converted numerical columns to proper numeric format and handled missing or invalid values.

7. **Feature Scaling**  
   Applied standardization to numerical features to improve the performance of machine learning models.

8. **Model Training**  
   Trained multiple classification models including Logistic Regression and Support Vector Machine (SVM).

9. **Model Evaluation**  
   Evaluated model performance using confusion matrix and accuracy score.



## Results

The performance of the models was evaluated using accuracy score and confusion matrix.

- When the test size was set to **0.2**:
  - **Logistic Regression** achieved an accuracy of **79%**
  - **Support Vector Machine (SVM)** achieved an accuracy of **80%**

- When the test size was increased to **0.3**:
  - **Logistic Regression** achieved an accuracy of **80%**
  - **Support Vector Machine (SVM)** achieved an accuracy of **79%**

This variation in accuracy occurs due to differences in data distribution between training and testing splits. It highlights the importance of data splitting and how model performance can slightly change depending on the sample used for evaluation.



## Model Selection

Multiple machine learning models were experimented with for this project, including:

- Decision Tree  
- Random Forest  
- Support Vector Machine (with different kernels)  
- Logistic Regression  

After comparing their performance, **Logistic Regression** and **Support Vector Machine (Linear Kernel)** were selected for final evaluation.

### Why Logistic Regression?
- Logistic Regression is a **strong baseline model** for binary classification problems.
- It is **simple, fast, and easy to understand**, making it suitable for initial modeling.
- It showed **stable accuracy** on the dataset across different train-test splits.

### Why Support Vector Machine (SVM)?
- SVM works well with **high-dimensional data**, which is created after one-hot encoding.
- The **linear kernel** was sufficient for this dataset and gave good results.
- SVM achieved **accuracy comparable to or slightly better than other models** in experiments.

### Reason for Choosing These Two Models
Although other models such as Decision Tree, Random Forest, and kernel-based SVMs were tested, **Logistic Regression and Linear SVM provided the best accuracy and consistency** among the models tried.  
Therefore, these two models were selected for the final comparison in this project.

## How to Run the Project

```bash
# Clone the repository
git clone https://github.com/Adityaraj1005/Telcom-Churn-Prediction.git
cd Telcom-Churn-Prediction

# Install required libraries
pip install numpy pandas matplotlib scikit-learn

# Run the project
python project.py 

```
## Results

The performance of the models was evaluated using **Accuracy** and **Confusion Matrix**.

Different train-test split ratios were tested to observe how the model behaves with changes in training data.

### Results Summary

- When **test size = 0.2**:
  - Logistic Regression achieved **79% accuracy**
  - Support Vector Machine achieved **80% accuracy**

- When **test size = 0.3**:
  - Logistic Regression achieved **80% accuracy**
  - Support Vector Machine achieved **79% accuracy**

This shows that a small change in the train-test split can slightly affect the accuracy of machine learning models.

Confusion matrices were also used to analyze the number of correct and incorrect predictions made by each model.



## Conclusion

In this project, a machine learning approach was used to predict customer churn using the Telco Customer Churn dataset.

The dataset was preprocessed by handling missing values, encoding categorical variables, and applying feature scaling where required.  
Two classification models — **Logistic Regression** and **Support Vector Machine (Linear Kernel)** — were trained and evaluated.

Both models produced **similar and consistent accuracy results**, with performance changing slightly depending on the train-test split ratio.  
This indicates that **no single model is always better**, and results can vary based on data distribution.

Overall, the project demonstrates how proper data preprocessing and model selection play an important role in building effective machine learning models for real-world problems like customer churn prediction.




## Problems Faced and Solutions

### Issue: Could not convert string to float error

While training the models, an error occurred stating that string values could not be converted to float.  
This happened because some numerical columns (such as `TotalCharges`) contained empty spaces or non-numeric values after encoding.

### Solution

The issue was resolved by:
- Converting the affected columns to numeric values using pandas
- Replacing invalid or missing values with `0`
- Applying feature scaling after cleaning the data

This ensured that all features passed to the machine learning models were numerical and compatible with the algorithms.


## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn