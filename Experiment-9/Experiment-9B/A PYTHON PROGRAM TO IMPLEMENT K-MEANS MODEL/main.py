import pandas as pd
import numpy as np
from math import sqrt

# --- 1. Load and Prepare Data ---
try:
    # Try loading from the path provided
    data = pd.read_csv('../input/k-means-clustering/KNN (3).csv')
except FileNotFoundError:
    print("Warning: Could not find dataset at '../input/k-means-clustering/KNN (3).csv'.")
    print("Please make sure 'KNN (3).csv' is in the correct directory.")
    # As a fallback, create dummy data to prevent errors.
    data = pd.DataFrame(np.random.rand(100, 5), columns=['col1', 'X1', 'X2', 'X3', 'Y'])
    data['Y'] = np.random.randint(0, 2, 100)

print("--- Initial Data Head (First 5 rows) ---")
print(data.head(5))

# Assume the first column is an ID, so we skip it
req_data = data.iloc[:, 1:]
print("\n--- Data for Use (Skipping first column) ---")
print(req_data.head(5))

# --- 2. Shuffle and Split Data ---
# Shuffling the row index of our dataset
np.random.seed(42) # for reproducibility
shuffle_index = np.random.permutation(req_data.shape[0])
req_data = req_data.iloc[shuffle_index]

# Splitting into train and test sets (70% train, 30% test)
train_size = int(req_data.shape[0] * 0.7)
train_df = req_data.iloc[:train_size, :]
test_df = req_data.iloc[train_size:, :]

# Convert to numpy arrays for efficiency
train = train_df.values
test = test_df.values
y_true = test[:, -1] # Get the true labels from the test set

print('\n--- Data Shapes ---')
print('Train_Shape: ', train_df.shape)
print('Test_Shape: ', test_df.shape)


# --- 3. KNN Core Functions ---

def euclidean_distance(x_test, x_train):
    """
    Calculates the Euclidean distance between two data points (rows).
    Ignores the last column (which is the label).
    """
    distance = 0
    for i in range(len(x_test) - 1): # Iterate through features, stop before label
        distance += (x_test[i] - x_train[i])**2
    return sqrt(distance)

def get_neighbors(x_test_row, x_train_data, num_neighbors):
    """
    Finds the 'num_neighbors' closest training samples to a single test sample.
    """
    distances = []
    data_rows = []
    
    for train_row in x_train_data:
        dist = euclidean_distance(x_test_row, train_row)
        distances.append(dist)
        data_rows.append(train_row)
        
    distances = np.array(distances)
    data_rows = np.array(data_rows)
    
    # argsort() function returns indices that would sort the array
    sort_indexes = distances.argsort()
    
    # Reorder the training data rows based on sorted distances
    sorted_data_rows = data_rows[sort_indexes]
    
    # Return the top 'num_neighbors' closest rows
    return sorted_data_rows[:num_neighbors]

def prediction(x_test_row, x_train_data, num_neighbors):
    """
    Predicts the class for a single test sample based on its neighbors.
    """
    classes = []
    neighbors = get_neighbors(x_test_row, x_train_data, num_neighbors)
    
    # Get the class label (last element) from each neighbor
    for neighbor in neighbors:
        classes.append(neighbor[-1])
        
    # Find the most frequent class among the neighbors
    predicted = max(set(classes), key=classes.count)
    return predicted

def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of the predictions.
    """
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            num_correct += 1
    
    accuracy_score = num_correct / len(y_true)
    return accuracy_score

# --- 4. Run Predictions and Calculate Accuracy ---
print("\n--- Running Predictions (k=5) ---")
k = 5
y_pred = []
for test_row in test:
    y_pred.append(prediction(test_row, train, k))

print("First 10 Predictions: ", y_pred[:10])
print("First 10 True Labels: ", y_true[:10].tolist())

# Calculate and print final accuracy
final_accuracy = accuracy(y_true, y_pred)
print(f"\n--- Final Accuracy (k={k}) ---")
print(f"Accuracy: {final_accuracy * 100:.2f}%")

