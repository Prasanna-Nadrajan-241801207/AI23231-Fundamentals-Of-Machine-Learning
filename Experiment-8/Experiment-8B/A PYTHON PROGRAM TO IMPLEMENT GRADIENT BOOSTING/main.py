import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree

# --- Part 1: Manual Step-by-Step Demonstration ---

print("--- Part 1: Manual Boosting Steps ---")
np.random.seed(42)
X_df = np.random.rand(100, 1) - 0.5
y_df = 3 * X_df[:, 0]**2 + 0.05 * np.random.randn(100)
df = pd.DataFrame()
df['X'] = X_df.reshape(100)
df['y'] = y_df

# Plot initial data
plt.figure(figsize=(10, 6))
plt.scatter(df['X'], df['y'])
plt.title('X vs y (Initial Data)')
plt.show()

# --- Step 1: Predict with the mean ---
df['pred1'] = df['y'].mean()
df['res1'] = df['y'] - df['pred1']
print(f"Initial Mean (Prediction 1): {df['pred1'].iloc[0]}")

# Plot first prediction
plt.figure(figsize=(10, 6))
plt.scatter(df['X'], df['y'], label='Data')
plt.plot(df['X'], df['pred1'], color='red', label=f"Prediction 1 (Mean = {df['pred1'].iloc[0]:.2f})")
plt.title('Step 1: Prediction with Mean')
plt.legend()
plt.show()

# --- Step 2: Fit a tree to the residuals ---
tree1 = DecisionTreeRegressor(max_leaf_nodes=8, random_state=42)
tree1.fit(df['X'].values.reshape(100, 1), df['res1'].values)

# Plot the first tree
plt.figure(figsize=(16, 10))
plot_tree(tree1, filled=True, feature_names=['X'])
plt.title("First Decision Tree (Fitted on Residuals 'res1')")
plt.show()

# --- Step 3: Create new prediction and residuals ---
# New prediction = old prediction (mean) + learning_rate * tree_prediction
# (Assuming learning_rate = 1 here for simplicity, as in the user's code)
df['pred2'] = df['pred1'] + tree1.predict(df['X'].values.reshape(100, 1))
df['res2'] = df['y'] - df['pred2']

print("\n--- Part 2: Automated Gradient Boosting Function ---")

def gradient_boost(X, y, number, lr, count=1, regs=None, foo=None):
    """
    Recursive function to perform gradient boosting.
    
    :param X: Feature matrix
    :param y: Target values (or residuals from previous step)
    :param number: Number of boosting stages to perform
    :param lr: Learning rate
    :param count: Internal counter for recursion
    :param regs: List to store the trained regressors
    :param foo: The original y values (for plotting)
    """
    if number == 0:
        print("Boosting complete.")
        return
    else:
        # Initialize list and original y on first call
        if regs is None:
            regs = []
        if foo is None:
            foo = y
        
        # If count > 1, we fit on the *new* residuals
        # Note: The provided logic fits on (y - last_prediction).
        # A more standard approach fits on the negative gradient, 
        # which for MSE is just the residuals (y - last_prediction).
        if count > 1:
            # Calculate new residuals
            y_res = foo - sum(lr * reg.predict(X) for reg in regs)
        else:
            # First time, fit on original y (or mean residuals)
            # The user's code fits on the original 'y' in the automated part.
            y_res = y 
            
        # Fit a new tree on the current target/residuals
        tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
        tree_reg.fit(X, y_res)
        
        # Add the new tree to our list
        regs.append(tree_reg)
        
        # --- Plotting the current state ---
        x1 = np.linspace(-0.5, 0.5, 500)
        
        # Calculate the combined prediction from *all* trees so far
        y_pred = sum(lr * regressor.predict(x1.reshape(-1, 1)) for regressor in regs)
        
        print(f"Iteration: {count}, Trees: {len(regs)}")
        plt.figure(figsize=(10, 6))
        plt.plot(x1, y_pred, "g-", linewidth=2, label=f'Prediction (n_trees={count})')
        plt.plot(X[:, 0], foo, "b.", markersize=6, label='Data')
        plt.plot(x1, 3*x1**2, "r--", label="True Function") # Plot true function
        plt.title(f"Gradient Boosting - Iteration {count}")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.show()
        
        # Recursive call for the next boosting stage
        gradient_boost(X, y, number - 1, lr, count + 1, regs=regs, foo=foo)


# --- Main Execution for Automated Function ---
print("\n--- Running Automated Gradient Boost ---")
np.random.seed(42)
X_auto = np.random.rand(100, 1) - 0.5
y_auto = 3 * X_auto[:, 0]**2 + 0.05 * np.random.randn(100)

# Run 5 stages of gradient boosting with a learning rate of 1.0
gradient_boost(X_auto, y_auto, 5, lr=1.0)

