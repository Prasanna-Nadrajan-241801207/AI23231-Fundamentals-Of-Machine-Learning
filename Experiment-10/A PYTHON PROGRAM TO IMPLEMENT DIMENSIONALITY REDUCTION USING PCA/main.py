from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt # Import for plotting

# --- 1. Load Iris Data ---
iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("--- Original Data (First 5 rows) ---")
print(df.head())

# --- 2. Scale the Data ---
# PCA is affected by scale, so we scale the features
scalar = StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(df))
print("\n--- Scaled Data (First 5 rows) ---")
print(scaled_data.head())

# --- 3. Correlation Heatmap (Before PCA) ---
print("\nPlotting Correlation Heatmap (Before PCA)...")
plt.figure(figsize=(8, 6))
sns.heatmap(scaled_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Scaled Features (Before PCA)")
plt.show()

# --- 4. Perform PCA ---
# Reduce from 4 features to 3 principal components
pca = PCA(n_components=3)
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)

# Create a DataFrame for the PCA results
data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])
print("\n--- PCA Transformed Data (First 5 rows) ---")
print(data_pca.head())

# --- 5. Correlation Heatmap (After PCA) ---
# The principal components should be uncorrelated
print("\nPlotting Correlation Heatmap (After PCA)...")
plt.figure(figsize=(8, 6))
sns.heatmap(data_pca.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Principal Components (After PCA)")
plt.show()

print("\n--- Explained Variance Ratio ---")
print(f"Explained variance by each component: {pca.explained_variance_ratio_}")
print(f"Total variance explained by 3 components: {np.sum(pca.explained_variance_ratio_):.2f}")

