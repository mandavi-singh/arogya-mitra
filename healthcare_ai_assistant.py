import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib
import pickle
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Create models folder
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv('dataset.csv')
desc_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")
severity_df = pd.read_csv("Symptom-severity.csv")

# Fill missing values
df.fillna("None", inplace=True)

# Drop duplicate rows to reduce overfitting
df.drop_duplicates(subset=df.columns.tolist(), inplace=True)
counts = df['Disease'].value_counts()
df = df[df['Disease'].isin(counts[counts > 2].index)]

# Prepare symptoms
symptom_columns = df.columns[:-1]
all_symptoms = sorted(set(sym for col in symptom_columns for sym in df[col].unique() if sym != "None"))

# Save all symptoms
joblib.dump(all_symptoms, 'models/all_symptoms.pkl')

# Encode symptoms
def encode_symptoms(row):
    symptoms_set = set(row)
    return [1 if symptom in symptoms_set else 0 for symptom in all_symptoms]

X = df[symptom_columns].apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms
y = df['Disease']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Train Calibrated SVM
svm_model = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True)
cal_svm = CalibratedClassifierCV(svm_model, cv=3)
cal_svm.fit(X_train, y_train)
svm_pred = cal_svm.predict(X_test)

# Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

print("\nRandom Forest Report:\n", classification_report(y_test, rf_pred, target_names=le.classes_))
print("SVM Report:\n", classification_report(y_test, svm_pred, target_names=le.classes_))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("RF CV Accuracy:", cross_val_score(rf_model, X, y_encoded, cv=cv).mean())
print("SVM CV Accuracy:", cross_val_score(cal_svm, X, y_encoded, cv=cv).mean())

# Hyperparameter tuning (optional)
params = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(probability=True), param_grid=params, cv=cv, scoring='accuracy')
grid.fit(X_train, y_train)
print("Best SVM params:", grid.best_params_)

# Save models
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(cal_svm, 'models/svm_model.pkl')

# Prediction function
def predict_disease(symptoms, model, all_symptoms, label_encoder):
    input_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    prediction = model.predict([input_vector])
    return label_encoder.inverse_transform(prediction)[0]

# Example test
test_symptoms = ['headache', 'fever']
print("\nPredicted disease (Random Forest):", predict_disease(test_symptoms, rf_model, all_symptoms, le))




