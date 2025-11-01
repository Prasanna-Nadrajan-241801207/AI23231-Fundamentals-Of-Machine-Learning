import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mlxtend.plotting import plot_decision_regions

# --- 1. Data Setup ---
df = pd.DataFrame()
df['X1'] = [1, 2, 3, 4, 5, 6, 6, 7, 9, 9]
df['X2'] = [5, 3, 6, 8, 1, 9, 5, 8, 9, 2]
# Original labels are {0, 1}. AdaBoost math is simpler with {-1, 1}.
df['label'] = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
df['label'] = df['label'].map({0: -1, 1: 1}) # Map 0->-1, 1->1

# Initial plot
print("--- Initial Data ---")
sns.scatterplot(x=df['X1'], y=df['X2'], hue=df['label'], palette='Set1')
plt.title("Initial Data Points")
plt.show()

# --- 2. Initial Weights ---
# Start with equal weights for all samples
df['weights'] = 1 / df.shape[0]

# Prepare data for sklearn
# Use .values to get numpy arrays
x = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values

# --- 3. Helper Functions ---

def calculate_model_weight(error):
    """Calculates the model weight (alpha) based on its error."""
    # Add a small epsilon to prevent division by zero if error is 0
    epsilon = 1e-10
    return 0.5 * np.log((1.0 - error + epsilon) / (error + epsilon))

def create_new_dataset(df_with_weights):
    """Resamples the dataset based on normalized weights."""
    df_with_weights['cumsum_upper'] = np.cumsum(df_with_weights['normalized_weights'])
    df_with_weights['cumsum_lower'] = df_with_weights['cumsum_upper'] - df_with_weights['normalized_weights']
    
    indices = []
    for i in range(df_with_weights.shape[0]):
        a = np.random.random()
        for index, row in df_with_weights.iterrows():
            if row['cumsum_upper'] > a and a > row['cumsum_lower']:
                indices.append(index)
    
    # Return the resampled data (features and labels)
    return df_with_weights.iloc[indices, [0, 1, 2]]


# --- 4. AdaBoost Iterations ---
# We will store the models (classifiers) and their weights (alphas)
classifiers = []
alphas = []
N_ESTIMATORS = 3 # Let's train 3 weak learners

# Make a copy of the dataframe to track weights
df_weights = df.copy()

for i in range(N_ESTIMATORS):
    print(f"\n--- Training Model {i+1} ---")
    
    # --- Step 1: Train Model on (re)sampled data ---
    if i == 0:
        # For the first model, train on the original data
        resampled_data = df_weights.iloc[:, [0, 1, 2]]
    else:
        # For subsequent models, resample based on last weights
        resampled_data = create_new_dataset(df_weights)
    
    x_resampled = resampled_data.iloc[:, 0:2].values
    y_resampled = resampled_data.iloc[:, 2].values
    
    # Create a new weak learner (a "stump" or shallow tree)
    dt = DecisionTreeClassifier(max_depth=1)
    dt.fit(x_resampled, y_resampled)
    
    # --- Step 2: Calculate Error and Alpha on *original* data ---
    y_pred = dt.predict(x)
    
    # Calculate weighted error
    error = np.sum(df_weights['weights'][df_weights['label'] != y_pred])
    
    # Calculate model weight (alpha)
    alpha = calculate_model_weight(error)
    
    # Store the classifier and its alpha
    classifiers.append(dt)
    alphas.append(alpha)
    
    print(f"Model {i+1} Error: {error:.4f}, Alpha: {alpha:.4f}")
    
    # Plot decision boundary for this model
    plot_decision_regions(x, y, clf=dt, legend=2)
    plt.title(f"Model {i+1} (Alpha: {alpha:.2f})")
    plt.show()

    # --- Step 3: Update Weights for *next* iteration ---
    # new_weight = old_weight * exp(-alpha * y_true * y_pred)
    df_weights['updated_weights'] = df_weights['weights'] * np.exp(-alpha * df_weights['label'] * y_pred)
    
    # Normalize the new weights so they sum to 1
    df_weights['normalized_weights'] = df_weights['updated_weights'] / df_weights['updated_weights'].sum()
    
    # Set the 'weights' for the next loop
    df_weights['weights'] = df_weights['normalized_weights']

print("\n--- Final Model Alphas ---")
print(alphas)


# --- 5. Final Prediction ---
def predict_adaboost(query, models, model_alphas):
    """Makes a final prediction by combining all weak learners."""
    final_vote = 0
    predictions = []
    
    for dt, alpha in zip(models, model_alphas):
        pred = dt.predict(query)[0]
        predictions.append(pred)
        final_vote += alpha * pred
        
    print(f"  Query: {query.flatten()}, Preds: {predictions}")
    print(f"  Final Vote: {final_vote:.4f} -> Sign: {np.sign(final_vote)}")
    return np.sign(final_vote)

print("\n--- Making Final Predictions ---")

# Query 1
query1 = np.array([1, 5]).reshape(1, 2)
final_pred1 = predict_adaboost(query1, classifiers, alphas)
print(f"  Final Prediction for {query1.flatten()}: {final_pred1}")

# Query 2
query2 = np.array([9, 9]).reshape(1, 2)
final_pred2 = predict_adaboost(query2, classifiers, alphas)
print(f"  Final Prediction for {query2.flatten()}: {final_pred2}")

# Query 3 (Example)
query3 = np.array([6, 5]).reshape(1, 2)
final_pred3 = predict_adaboost(query3, classifiers, alphas)
print(f"  Final Prediction for {query3.flatten()}: {final_pred3}")

