import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
try:
    data = pd.read_csv('headbrain.csv')
except FileNotFoundError:
    print("Error: 'headbrain.csv' not found.")
    print("Please make sure the dataset file is in the same directory.")
    exit()

x, y = np.array(list(data['Head Size(cm^3)'])), np.array(list(data['Brain Weight(grams)']))
print(x[:5], y[:5])
# Output: [4512 3738 4261 3777 4177] [1530 1297 1335 1282 1590]

def get_line(x, y):
    """
    Calculates the slope (m) and intercept (c) for the line of best fit.
    """
    x_m, y_m = np.mean(x), np.mean(y)
    print(f"Mean Head Size: {x_m}")
    print(f"Mean Brain Weight: {y_m}")
    # Output: 3609.389... 1282.873...
    
    x_d, y_d = x - x_m, y - y_m
    
    # Calculate slope (m)
    m = np.sum(x_d * y_d) / np.sum(x_d**2)
    # Calculate intercept (c)
    c = y_m - (m * x_m)
    
    print(f"Slope (m): {m}")
    print(f"Intercept (c): {c}")
    # Output: 0.263... 325.573...
    
    # Return a function for the line
    return lambda x_val : m * x_val + c

# Get the regression line function
lin = get_line(x, y)

# --- Plotting the Regression Line ---
# Create a range of x values for plotting the line
X_plot = np.linspace(np.min(x) - 100, np.max(x) + 100, 1000)
# Calculate the corresponding y values using the line function
Y_plot = np.array([lin(x_val) for x_val in X_plot])

plt.figure(figsize=(10, 6))
plt.plot(X_plot, Y_plot, color='red', label='Regression line')
plt.scatter(x, y, color='green', label='Scatter plot')
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.title('Head Size vs Brain Weight (Linear Regression from Scratch)')
plt.grid(True)
plt.show()

def get_error(line_func, x, y):
    """
    Calculates the R-squared (coefficient of determination) error.
    """
    y_m = np.mean(y)
    y_pred = np.array([line_func(x_val) for x_val in x])
    
    # Total sum of squares
    ss_t = np.sum((y - y_m)**2)
    # Residual sum of squares
    ss_r = np.sum((y - y_pred)**2)
    
    # R-squared value
    return 1 - (ss_r / ss_t)

r_squared = get_error(lin, x, y)
print(f"\nR-squared (from scratch): {r_squared}")
# Output: 0.639311...

# --- Comparison with In-built Package (scikit-learn) ---
print("\n--- Comparing with scikit-learn ---")
from sklearn.linear_model import LinearRegression

# sklearn requires X to be a 2D array
x_reshaped = x.reshape((len(x), 1))

reg = LinearRegression()
reg = reg.fit(x_reshaped, y)

score = reg.score(x_reshaped, y)
print(f"R-squared (from sklearn): {score}")
# Output: 0.639311...

print(f"\nSlope (sklearn): {reg.coef_[0]}")
print(f"Intercept (sklearn): {reg.intercept_}")

