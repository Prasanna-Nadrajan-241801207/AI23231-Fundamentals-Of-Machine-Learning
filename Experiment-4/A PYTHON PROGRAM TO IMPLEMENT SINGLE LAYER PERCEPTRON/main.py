import numpy as np
import pandas as pd

# Define the input for the AND gate
# (0,0), (0,1), (1,1), (1,0)
input_value = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
print(f"Input shape: {input_value.shape}")

# Define the output for the AND gate
# 0 AND 0 = 0
# 0 AND 1 = 0
# 1 AND 1 = 1
# 1 AND 0 = 0
output = np.array([0, 0, 1, 0])
# Reshape to a column vector
output = output.reshape(4, 1)
print(f"Output shape: {output.shape}")

# Initialize weights and bias
weights = np.array([[0.1], [0.3]])
bias = 0.2
print(f"\nInitial Weights: \n{weights}")
print(f"Initial Bias: {bias}")

# Learning rate
learning_rate = 0.05
# Number of epochs for training
epochs = 15000

def sigmoid_func(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def der(x):
    """Derivative of the sigmoid function"""
    return sigmoid_func(x) * (1 - sigmoid_func(x))

print("\nStarting training...")

for i in range(epochs):
    # --- Forward Propagation ---
    
    # 1. Calculate weighted sum (z)
    weighted_sum = np.dot(input_value, weights) + bias
    
    # 2. Apply activation function
    first_output = sigmoid_func(weighted_sum)
    
    # --- Backpropagation (Gradient Descent) ---
    
    # 1. Calculate the error (predicted - actual)
    error = first_output - output
    
    # (Optional: Calculate total error for monitoring)
    if i % 1000 == 0:
        total_error = np.square(np.subtract(first_output, output)).mean()
        # print(f"Epoch {i}, Error: {total_error}")

    # 2. Calculate the derivative of the error
    # d(Error)/d(Prediction) = error
    first_der = error 
    
    # 3. Calculate the derivative of the prediction
    # d(Prediction)/d(z) = sigmoid_derivative(prediction)
    second_der = der(first_output)
    
    # 4. Combine derivatives (Chain Rule)
    # d(Error)/d(z)
    derivative = first_der * second_der
    
    # 5. Calculate derivative with respect to weights
    # d(z)/d(w) = input
    # d(Error)/d(w) = d(Error)/d(z) * d(z)/d(w)
    t_input = input_value.T
    final_derivative_weights = np.dot(t_input, derivative)
    
    # 6. Update weights
    weights = weights - (learning_rate * final_derivative_weights)
    
    # 7. Update bias
    # d(Error)/d(b) = d(Error)/d(z) * d(z)/d(b)
    # d(z)/d(b) = 1
    # We sum the derivatives for the bias across all samples in the batch
    final_derivative_bias = np.sum(derivative)
    bias = bias - (learning_rate * final_derivative_bias)

print("Training complete.")
print(f"\nTrained Weights: \n{weights}")
print(f"Trained Bias: \n{bias}")
# Expected output:
# [[16.57...], [16.57...]]
# [-25.14...]

print("\n--- Making Predictions ---")

def predict(x1, x2):
    pred_input = np.array([x1, x2])
    result = np.dot(pred_input, weights) + bias
    res = sigmoid_func(result)
    return res[0]

# Test [1, 0] -> 0
res_1_0 = predict(1, 0)
print(f"Prediction for [1, 0]: {res_1_0} (Expected ~0)")

# Test [1, 1] -> 1
res_1_1 = predict(1, 1)
print(f"Prediction for [1, 1]: {res_1_1} (Expected ~1)")

# Test [0, 0] -> 0
res_0_0 = predict(0, 0)
print(f"Prediction for [0, 0]: {res_0_0} (Expected ~0)")

# Test [0, 1] -> 0
res_0_1 = predict(0, 1)
print(f"Prediction for [0, 1]: {res_0_1} (Expected ~0)")

