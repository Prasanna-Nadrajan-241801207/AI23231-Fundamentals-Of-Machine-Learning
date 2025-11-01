import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --- 1. Load Data ---
iris = load_iris()

# --- 2. Set Parameters ---
n_classes = 3
plot_colors = "ryb" # red, yellow, blue
plot_step = 0.02

# --- 3. Plot Decision Boundaries for Feature Pairs ---

# Set up the figure for subplots
plt.figure(figsize=(16, 10))

# List of feature pairs
pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

for pairidx, pair in enumerate(pairs):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train the classifier
    clf = DecisionTreeClassifier().fit(X, y)

    # Create a subplot for this pair
    plt.subplot(2, 3, pairidx + 1)

    # Find the min/max for the plot grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create the meshgrid
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), 
        np.arange(y_min, y_max, plot_step)
    )
    
    # Adjust layout
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # Predict the class for each point in the meshgrid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundaries (filled contours)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    # Add labels
    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15
        )

# Add a title and legend to the overall figure
plt.suptitle("Decision surface of decision trees trained on pairs of features", fontsize=16)
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
plt.axis("tight")

# --- 4. Plot Full Decision Tree (All Features) ---

plt.figure(figsize=(20, 15))
clf_full = DecisionTreeClassifier().fit(iris.data, iris.target)

plot_tree(
    clf_full,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True
)

plt.title("Decision tree trained on all the iris features", fontsize=20)
plt.show()

