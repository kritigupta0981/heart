import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# 1. Load Data
DATA_PATH = 'heart disease.csv'
if not os.path.exists(DATA_PATH):
    # Try with underscore if space fails
    DATA_PATH = 'heart_disease.csv'

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape}")

# 2. Data Cleaning
df = df.drop_duplicates()
print(f"Dataset after removing duplicates: {df.shape}")

# 3. Simple EDA (Correlation)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
print("Saved correlation_heatmap.png")

# 4. Preprocessing
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for use in the app
joblib.dump(scaler, 'scaler.pkl')
print("Saved scaler.pkl")

# 5. Model Training & Evaluation
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=2),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# 6. Comparison
comparison_df = pd.DataFrame(results)
print("\nModel Comparison Table:")
print(comparison_df)

# 7. Save Best Model (Random Forest usually performs best)
best_model = models['Random Forest']
joblib.dump(best_model, 'heart_model.pkl')
print("\nSaved Random Forest model as heart_model.pkl")

# 8. Sample Prediction
sample_data = pd.DataFrame({
    'age': [52], 'sex': [1], 'cp': [0], 'trestbps': [125], 'chol': [212],
    'fbs': [0], 'restecg': [1], 'thalach': [168], 'exang': [0],
    'oldpeak': [1.0], 'slope': [2], 'ca': [2], 'thal': [3]
})
sample_scaled = scaler.transform(sample_data)
prediction = best_model.predict(sample_scaled)
result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
print(f"\nSample Prediction for data:\n{sample_data}")
print(f"Result: {result}")
