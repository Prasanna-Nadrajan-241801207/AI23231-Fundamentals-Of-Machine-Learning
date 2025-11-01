import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# --- 1. Load Data ---
try:
    # Try loading from the path provided
    dataset = pd.read_csv('../input/mall-customers/Mall_Customers.csv')
except FileNotFoundError:
    print("Warning: Could not find dataset at '../input/mall-customers/Mall_Customers.csv'.")
    print("Please make sure 'Mall_Customers.csv' is in the correct directory.")
    # As a fallback, create dummy data to prevent errors,
    # though the plots will be meaningless.
    data = np.random.rand(100, 2) * 100
    X = np.hstack((data, np.random.randint(1, 100, (100, 1))))
    dataset = pd.DataFrame(X, columns=['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

# We are interested in 'Annual Income' and 'Spending Score'
X = dataset.iloc[:, [3, 4]].values

print("--- Dataset Head ---")
print(dataset.head())


# --- 2. Use the Elbow Method to find the optimal number of clusters ---
print("\nCalculating WCSS for Elbow Method...")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # inertia_ is the WCSS

# Plot the graph to visualize the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()


# --- 3. Apply K-Means with the optimal number of clusters (k=5) ---
print("Applying K-Means with k=5 clusters...")
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

print("\nCluster assignments (y_kmeans):")
print(y_kmeans)


# --- 4. Visualize the clusters ---
print("Plotting clusters...")
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

for i in range(5):
    plt.scatter(
        X[y_kmeans == i, 0],  # X-coordinate for points in cluster i
        X[y_kmeans == i, 1],  # Y-coordinate for points in cluster i
        s=100, 
        c=colors[i], 
        label=labels[i]
    )

# Plot the centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0], 
    kmeans.cluster_centers_[:, 1], 
    s=300, 
    c='yellow', 
    label='Centroids',
    edgecolors='black'
)

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

