import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
# Note: The provided path was '../input/iris-dataset/iris.csv'
# I've changed it to 'iris.csv' to be consistent with other experiments.
# This CSV file should have columns: 
# 'sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety'
try:
    df = pd.read_csv('iris.csv')
except FileNotFoundError:
    print("Error: 'iris.csv' not found.")
    print("Please make sure the Iris dataset file is in the same directory.")
    exit()

print("Dataset head:")
print(df.head())

print("\nDataset shape:")
print(df.shape)
print("\n")

# --- Univariate Analysis ---

print("Starting Univariate Analysis plots...")

# Define dataframes for each variety
df_Setosa = df.loc[df['variety'] == 'Setosa']
df_Virginica = df.loc[df['variety'] == 'Virginica']
df_Versicolor = df.loc[df['variety'] == 'Versicolor']

# 1. Univariate for sepal width
plt.figure(figsize=(10, 2))
plt.scatter(df_Setosa['sepal.width'], np.zeros_like(df_Setosa['sepal.width']), label='Setosa')
plt.scatter(df_Virginica['sepal.width'], np.zeros_like(df_Virginica['sepal.width']), label='Virginica')
plt.scatter(df_Versicolor['sepal.width'], np.zeros_like(df_Versicolor['sepal.width']), label='Versicolor')
plt.xlabel('sepal.width')
plt.title('Univariate Analysis: Sepal Width')
plt.yticks([]) # Hide y-axis ticks
plt.legend()
plt.grid(axis='x')
plt.show()

# 2. Univariate for sepal length
plt.figure(figsize=(10, 2))
plt.scatter(df_Setosa['sepal.length'], np.zeros_like(df_Setosa['sepal.length']), label='Setosa')
plt.scatter(df_Virginica['sepal.length'], np.zeros_like(df_Virginica['sepal.length']), label='Virginica')
plt.scatter(df_Versicolor['sepal.length'], np.zeros_like(df_Versicolor['sepal.length']), label='Versicolor')
plt.xlabel('sepal.length')
plt.title('Univariate Analysis: Sepal Length')
plt.yticks([]) # Hide y-axis ticks
plt.legend()
plt.grid(axis='x')
plt.show()

# 3. Univariate for petal width
plt.figure(figsize=(10, 2))
plt.scatter(df_Setosa['petal.width'], np.zeros_like(df_Setosa['petal.width']), label='Setosa')
plt.scatter(df_Virginica['petal.width'], np.zeros_like(df_Virginica['petal.width']), label='Virginica')
plt.scatter(df_Versicolor['petal.width'], np.zeros_like(df_Versicolor['petal.width']), label='Versicolor')
plt.xlabel('petal.width')
plt.title('Univariate Analysis: Petal Width')
plt.yticks([]) # Hide y-axis ticks
plt.legend()
plt.grid(axis='x')
plt.show()

# 4. Univariate for petal length
plt.figure(figsize=(10, 2))
plt.scatter(df_Setosa['petal.length'], np.zeros_like(df_Setosa['petal.length']), label='Setosa')
plt.scatter(df_Virginica['petal.length'], np.zeros_like(df_Virginica['petal.length']), label='Virginica')
plt.scatter(df_Versicolor['petal.length'], np.zeros_like(df_Versicolor['petal.length']), label='Versicolor')
plt.xlabel('petal.length')
plt.title('Univariate Analysis: Petal Length')
plt.yticks([]) # Hide y-axis ticks
plt.legend()
plt.grid(axis='x')
plt.show()


# --- Bivariate Analysis ---
print("\nStarting Bivariate Analysis plots...")

# 1. Bivariate sepal.width vs petal.width
# Note: 'size' is deprecated, using 'height' instead
sns.FacetGrid(df, hue='variety', height=5).map(plt.scatter, "sepal.width", "petal.width").add_legend()
plt.title('Bivariate Analysis: Sepal Width vs Petal Width')
plt.show()

# 2. Bivariate sepal.length vs petal.length
sns.FacetGrid(df, hue='variety', height=5).map(plt.scatter, "sepal.length", "petal.length").add_legend()
plt.title('Bivariate Analysis: Sepal Length vs Petal Length')
plt.show()


# --- Multivariate Analysis ---
print("\nStarting Multivariate Analysis (Pairplot)...")

# 1. Multivariate all the features
# Note: 'size' is deprecated, using 'height' instead
sns.pairplot(df, hue="variety", height=2)
plt.suptitle('Multivariate Analysis: Pairplot of Iris Features', y=1.02)
plt.show()

print("\nAll visualizations complete.")
