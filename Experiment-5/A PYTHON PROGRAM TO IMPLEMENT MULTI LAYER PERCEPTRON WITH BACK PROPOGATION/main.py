import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
try:
    bnotes = pd.read_csv('bank_note_data.csv')
    print("--- Dataset Head ---")
    print(bnotes.head(10))
except FileNotFoundError:
    print("Error: 'bank_note_data.csv' not found.")
    print("Please make sure the dataset file is in the same directory.")
    exit()

# Prepare data
x = bnotes.drop('Class', axis=1)
y = bnotes['Class']

print("\n--- Feature (x) Head ---")
print(x.head(2))
print("\n--- Target (y) Head ---")
print(y.head(2))

# --- Experiment 1: Test Size = 0.2 ---
print("\n" + "="*40)
print(" STARTING EXPERIMENT 1: test_size = 0.2")
print("="*40)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Activation: relu ---
print("\n--- Activation: relu (test_size=0.2) ---")
mlp_relu = MLPClassifier(max_iter=500, activation='relu', random_state=42)
mlp_relu.fit(x_train, y_train)
pred_relu = mlp_relu.predict(x_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_relu))
print("\nClassification Report:")
print(classification_report(y_test, pred_relu))

# --- Activation: logistic ---
print("\n--- Activation: logistic (test_size=0.2) ---")
mlp_logistic = MLPClassifier(max_iter=500, activation='logistic', random_state=42)
mlp_logistic.fit(x_train, y_train)
pred_logistic = mlp_logistic.predict(x_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_logistic))
print("\nClassification Report:")
print(classification_report(y_test, pred_logistic))

# --- Activation: tanh ---
print("\n--- Activation: tanh (test_size=0.2) ---")
mlp_tanh = MLPClassifier(max_iter=500, activation='tanh', random_state=42)
mlp_tanh.fit(x_train, y_train)
pred_tanh = mlp_tanh.predict(x_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_tanh))
print("\nClassification Report:")
print(classification_report(y_test, pred_tanh))

# --- Activation: identity ---
print("\n--- Activation: identity (test_size=0.2) ---")
mlp_identity = MLPClassifier(max_iter=500, activation='identity', random_state=42)
mlp_identity.fit(x_train, y_train)
pred_identity = mlp_identity.predict(x_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_identity))
print("\nClassification Report:")
print(classification_report(y_test, pred_identity))


# --- Experiment 2: Test Size = 0.3 ---
print("\n" + "="*40)
print(" STARTING EXPERIMENT 2: test_size = 0.3")
print("="*40)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# --- Activation: relu ---
print("\n--- Activation: relu (test_size=0.3) ---")
mlp_relu_30 = MLPClassifier(max_iter=500, activation='relu', random_state=42)
mlp_relu_30.fit(x_train, y_train)
pred_relu_30 = mlp_relu_30.predict(x_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_relu_30))
print("\nClassification Report:")
print(classification_report(y_test, pred_relu_30))

# --- Activation: logistic ---
print("\n--- Activation: logistic (test_size=0.3) ---")
mlp_logistic_30 = MLPClassifier(max_iter=500, activation='logistic', random_state=42)
mlp_logistic_30.fit(x_train, y_train)
pred_logistic_30 = mlp_logistic_30.predict(x_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_logistic_30))
print("\nClassification Report:")
print(classification_report(y_test, pred_logistic_30))

# --- Activation: tanh ---
print("\n--- Activation: tanh (test_size=0.3) ---")
mlp_tanh_30 = MLPClassifier(max_iter=500, activation='tanh', random_state=42)
mlp_tanh_30.fit(x_train, y_train)
pred_tanh_30 = mlp_tanh_30.predict(x_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_tanh_30))
print("\nClassification Report:")
print(classification_report(y_test, pred_tanh_30))

# --- Activation: identity ---
print("\n--- Activation: identity (test_size=0.3) ---")
mlp_identity_30 = MLPClassifier(max_iter=500, activation='identity', random_state=42)
mlp_identity_30.fit(x_train, y_train)
pred_identity_30 = mlp_identity_30.predict(x_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_identity_30))
print("\nClassification Report:")
print(classification_report(y_test, pred_identity_30))

