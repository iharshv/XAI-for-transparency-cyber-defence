# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Feature scaling
    scaler = StandardScaler()
    features = df.drop('label', axis=1)
    scaled_features = scaler.fit_transform(features)

    return scaled_features, df['label'], label_encoders, scaler


# src/train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump(model, 'models/trained_model.pkl')
    return model, X_test, y_test


# src/explain_model.py
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

def explain_with_shap(model_path, X_sample):
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample)


# src/utils.py
import os

def ensure_directories():
    os.makedirs('models', exist_ok=True)
    os.makedirs('screenshots', exist_ok=True)
    os.makedirs('diagrams', exist_ok=True)
