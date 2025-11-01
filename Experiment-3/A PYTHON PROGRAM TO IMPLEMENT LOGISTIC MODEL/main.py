import pandas as pd
import numpy as np
from numpy import log, dot, exp, shape
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
try:
    data = pd.read_csv('suv_data.csv')
    print("Dataset loaded successfully:")
    print(data.head())
except FileNotFoundError:
    print("Error: 'suv_data.csv' not found.")
    print("Please make sure the dataset file is in the same directory.")
    exit()

x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

# --- In-built Function (scikit-learn) ---
print("\n--- Using scikit-learn ---")

# 1. Split the data
x_train_sk, x_test_sk, y_train_sk, y_test_sk = train_test_split(x, y, test_size=0.10, random_state=0)

# 2. Feature Scaling
sc = StandardScaler()
x_train_sk = sc.fit_transform(x_train_sk)
x_test_sk = sc.transform(x_test_sk)

print("Scaled X_train (first 10 rows):")
print(x_train_sk[0:10, :])

# 3. Fit Logistic Regression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train_sk, y_train_sk)

# 4. Predict
y_pred_sk = classifier.predict(x_test_sk)
print(f"\nsklearn Predictions: {y_pred_sk}")

# 5. Evaluate
cm_sk = confusion_matrix(y_test_sk, y_pred_sk)
print("Confusion Matrix (sklearn): \n", cm_sk)

acc_sk = accuracy_score(y_test_sk, y_pred_sk)
print("Accuracy (sklearn): ", acc_sk)


# --- User Defined function (from scratch) ---
print("\n--- Using User-Defined Function from Scratch ---")

# 1. Split the data (using the same split for a fair comparison)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)

# Example 'Std' function (as provided)
def Std(input_data):
    mean0 = np.mean(input_data[:, 0])
    sd0 = np.std(input_data[:, 0])
    mean1 = np.mean(input_data[:, 1])
    sd1 = np.std(input_data[:, 1])
    return lambda x: ((x[0] - mean0) / sd0, (x[1] - mean1) / sd1)

my_std = Std(x)
print(f"\nDemo of Std function on one row: {my_std(x_train[0])}")

# Standardization function to be used
def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:, i] = (X_tr[:, i] - np.mean(X_tr[:, i])) / np.std(X_tr[:, i])

# F1 Score function
def F1_score(y, y_hat):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    
    # Add epsilon (1e-9) to avoid division by zero
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1_score = 2 * precision * recall / (precision + recall + 1e-9)
    return f1_score

# Logistic Regression Class
class MyLogisticRegression:
    def sigmoid(self, z):
        sig = 1 / (1 + exp(-z))
        return sig

    def initialize(self, X):
        weights = np.zeros((shape(X)[1] + 1, 1))
        # Add bias term (column of ones)
        X_with_bias = np.c_[np.ones((shape(X)[0], 1)), X]
        return weights, X_with_bias

    def fit(self, X, y, alpha=0.001, iter=400):
        weights, X_with_bias = self.initialize(X)
        
        def cost(theta):
            z = dot(X_with_bias, theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1 - y).T.dot(log(1 - self.sigmoid(z)))
            cost = -((cost1 + cost0)) / len(y)
            return cost
        
        cost_list = np.zeros(iter,)
        for i in range(iter):
            # Gradient descent
            z = dot(X_with_bias, weights)
            gradient = dot(X_with_bias.T, self.sigmoid(z) - np.reshape(y, (len(y), 1)))
            weights = weights - alpha * gradient
            cost_list[i] = cost(weights)
            
        self.weights = weights
        return cost_list

    def predict(self, X):
        # Get X with bias term
        X_with_bias = self.initialize(X)[1]
        z = dot(X_with_bias, self.weights)
        
        lis = []
        for i in self.sigmoid(z):
            if i > 0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis

# 2. Standardize the data
standardize(x_train)
standardize(x_test)

# 3. Fit the model
obj1 = MyLogisticRegression()
model = obj1.fit(x_train, y_train)

# 4. Predict
y_pred = obj1.predict(x_test)
y_trainn = obj1.predict(x_train)

# 5. Evaluate
f1_score_tr = F1_score(y_train, y_trainn)
f1_score_te = F1_score(y_test, y_pred)
print(f"\nF1 Score (Train, from scratch): {f1_score_tr}")
print(f"F1 Score (Test, from scratch): {f1_score_te}")

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (from scratch): \n", conf_mat)

# Calculate accuracy from scratch's confusion matrix
accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / sum(sum(conf_mat))
print("Accuracy (from scratch): ", accuracy)

