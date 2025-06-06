import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Load dataset
url = "https://raw.githubusercontent.com/Mohie-Aldin-Sinan/TuneMood/main/data/Acoustic%20Features.csv"
df = pd.read_csv(url)

# Prepare features and labels
X = df.drop(columns=['label'])
y = df['label']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model training
model = LogisticRegression(max_iter=1000, class_weight={'angry': 1, 'happy': 1, 'relax': 1, 'sad': 2})
model.fit(x_train_scaled, y_train)

# Evaluation
y_pred = model.predict(x_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

joblib.dump(model, "TuneMood_model.pkl")
joblib.dump(scaler, "TuneMood_scaler.pkl")
