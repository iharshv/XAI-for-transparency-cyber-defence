import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data: Replace this with your actual dataset
# This is just an example, make sure to load your own dataset here
data = {
    'src_bytes': [500, 1500, 200, 10000, 300, 50, 7000],
    'dst_bytes': [500, 1200, 100, 15000, 300, 100, 8000],
    'duration': [120, 300, 50, 2000, 80, 20, 1500],
    'flag': ['SF', 'SF', 'REJ', 'SF', 'SF', 'SF', 'RSTO'],  # example flags
    'protocol': ['TCP', 'TCP', 'UDP', 'TCP', 'UDP', 'TCP', 'TCP'],
    'malicious': [0, 1, 0, 1, 0, 0, 1]  # 0: benign, 1: malicious
}

# Load the data into a DataFrame
df = pd.DataFrame(data)

# Step 3: Preprocess the Data
# Convert categorical features ('flag' and 'protocol') into numerical format
df['flag'] = df['flag'].map({'SF': 1, 'REJ': 0, 'RSTO': 2})  # Example encoding
df['protocol'] = df['protocol'].map({'TCP': 1, 'UDP': 0})

# Define features (X) and target (y)
X = df.drop(columns=['malicious'])  # Features
y = df['malicious']  # Target variable (malicious or not)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (optional but often recommended for better performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the Model with Class Weights to Handle Class Imbalance
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))  # Use zero_division to suppress warning

# Step 8: Visualize Feature Importances (XAI)
feature_importances = model.feature_importances_

# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=feature_importances)
plt.title("Feature Importance for Malicious Connection Prediction")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

# Step 9: Classify New Connection (Example)
# Define a new connection in the same structure as your training data
new_connection = pd.DataFrame([[800, 1200, 300, 1, 1]], columns=['src_bytes', 'dst_bytes', 'duration', 'flag', 'protocol'])

# Ensure that the new connection is scaled using the same scaler fitted on the training data
new_connection_scaled = scaler.transform(new_connection)

# Predict using the trained model
prediction = model.predict(new_connection_scaled)

if prediction == 1:
    print("This connection is malicious.")
else:
    print("This connection is benign.")
'''
# Step 9: Classify New Connection (Example)
# Create a new connection that looks like a malicious connection
# For example, large src_bytes, dst_bytes, and duration might indicate malicious behavior
new_connection = pd.DataFrame([[5000, 20000, 1200, 2, 1]], columns=['src_bytes', 'dst_bytes', 'duration', 'flag', 'protocol'])

# Ensure the new connection is scaled using the same scaler fitted on the training data
new_connection_scaled = scaler.transform(new_connection)

# Predict using the trained model
prediction = model.predict(new_connection_scaled)

if prediction == 1:
    print("This connection is malicious.")
else:
    print("This connection is benign.")'''
