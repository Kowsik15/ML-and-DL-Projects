# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 2025

@author: KOWSIK.S
"""

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import os

# Set working directory (update this path as needed)
os.chdir(r"C:\Users\KOWSIK.S\Data_Science\ML\Capstone")

# Load dataset (replace with your actual dataset path)
df = pd.read_csv('personality_dataset.csv')

# Define features and target
X = df[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 
        'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']]
y = df['Personality']

# Preprocessing
# Impute missing values
numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']
categorical_columns = ['Stage_fear', 'Drained_after_socializing']

# Median imputation for numeric features
numeric_imputer = SimpleImputer(strategy='median')
X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])

# Mode imputation for categorical features
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])

# Encode categorical variables
le_stage_fear = LabelEncoder()
X['Stage_fear'] = le_stage_fear.fit_transform(X['Stage_fear'])

le_drained = LabelEncoder()
X['Drained_after_socializing'] = le_drained.fit_transform(X['Drained_after_socializing'])

le_personality = LabelEncoder()
y = le_personality.fit_transform(y)  # Introvert=1, Extrovert=0

# Scale numeric features
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
print("\n--- Random Forest Evaluation ---")
print("Training accuracy RF:", rf_model.score(X_train, y_train))
print("Testing accuracy RF:", rf_model.score(X_test, y_test))


# Train Logistic Regression
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Evaluate model
print("\n--- Logistic Regression Evaluation ---")
print("Training accuracy logreg:", log_reg_model.score(X_train, y_train))
print("Testing accuracy Logreg:", log_reg_model.score(X_test, y_test))

# Train KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Evaluate model
print("\n--- KNN Evaluation ---")
print("Training accuracy KNN:", knn_model.score(X_train, y_train))
print("Testing accuracy KNN:", knn_model.score(X_test, y_test))

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate model
print("\n--- Decsion Tree Evaluation ---")
print("Training accuracy DT:", dt_model.score(X_train, y_train))
print("Testing accuracy DT:", dt_model.score(X_test, y_test))

# Train SVM
svm_model = SVC(probability=True)  # probability=True if you want to get prediction probabilities
svm_model.fit(X_train, y_train)

# Evaluate model
print("\n--- SVM Evaluation ---")
print("Training accuracy SVM:", svm_model.score(X_train, y_train))
print("Testing accuracy SVM:", svm_model.score(X_test, y_test))

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
# Predict on training and test sets
xgb_train_preds = xgb_model.predict(X_train)
xgb_test_preds = xgb_model.predict(X_test)

# Print evaluation results
print("\n--- XGBoost Evaluation ---")
print("Training Accuracy:", accuracy_score(y_train, xgb_train_preds))
print("Testing Accuracy:", accuracy_score(y_test, xgb_test_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_test_preds))
print("Classification Report:\n", classification_report(y_test, xgb_test_preds, target_names=['Extrovert', 'Introvert']))

# Train AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)

print("\n--- ADABoost Evaluation ---")
print("Training accuracy ADA:", ada_model.score(X_train, y_train))
print("Testing accuracy ADA:", ada_model.score(X_test, y_test))

# Save model and preprocessors
os.makedirs('models', exist_ok=True)
pickle.dump(rf_model, open('models/model_rf.pkl', 'wb'))
pickle.dump(log_reg_model, open('models/model_logistic.pkl', 'wb'))
pickle.dump(knn_model, open('models/model_knn.pkl', 'wb'))
pickle.dump(dt_model, open('models/model_dt.pkl', 'wb'))
pickle.dump(svm_model, open('models/model_svm.pkl', 'wb'))
pickle.dump(xgb_model, open('models/model_xgb.pkl', 'wb'))
pickle.dump(ada_model, open('models/model_ada.pkl', 'wb'))


pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(le_stage_fear, open('models/label_encoder_stage_fear.pkl', 'wb'))
pickle.dump(le_drained, open('models/label_encoder_drained.pkl', 'wb'))

# Example prediction
sample = pd.DataFrame([[9, 'Yes', 0, 0, 'Yes', 0, 3]],
                      columns=['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                               'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                               'Post_frequency'])
sample[numeric_columns] = numeric_imputer.transform(sample[numeric_columns])
sample[categorical_columns] = categorical_imputer.transform(sample[categorical_columns])
sample['Stage_fear'] = le_stage_fear.transform(sample['Stage_fear'])
sample['Drained_after_socializing'] = le_drained.transform(sample['Drained_after_socializing'])
sample[numeric_columns] = scaler.transform(sample[numeric_columns])

prediction_rf = rf_model.predict(sample)[0]
prediction_log = log_reg_model.predict(sample)[0]
prediction_knn = knn_model.predict(sample)[0]
prediction_dt = dt_model.predict(sample)[0]
prediction_svm = svm_model.predict(sample)[0]
prediction_xgb = xgb_model.predict(sample)[0]
prediction_ada = ada_model.predict(sample)[0]




# Sample prediction from each model

for name, models in {
    "Logistic_Regression": log_reg_model,
    "Random_Forest": rf_model,
    "KNN": knn_model,
    "Decision_Tree": dt_model,
    "SVM": svm_model,
    "AdaBoost": ada_model,
    "XGBoost": xgb_model
}.items():
    prediction = models.predict(sample)[0]
    print('\n'f"Sample prediction {name}:", 'Introvert' if prediction == 1 else 'Extrovert')
