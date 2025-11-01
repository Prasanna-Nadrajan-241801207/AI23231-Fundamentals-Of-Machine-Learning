import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set seaborn style
sns.set(font_scale=1.2)

# Load the dataset
try:
    recipes = pd.read_csv('recipes_muffins_cupcakes.csv')
    print("--- Dataset Head ---")
    print(recipes.head())
    print(f"\nDataset shape: {recipes.shape}")
except FileNotFoundError:
    print("Error: 'recipes_muffins_cupcakes.csv' not found.")
    print("Please make sure the dataset file is in the same directory.")
    exit()

# Initial scatter plot
print("\n--- Plotting Sugar vs. Flour ---")
sns.lmplot(x='Sugar', y='Flour', data=recipes, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.title('Muffins vs. Cupcakes (Sugar vs. Flour)')
plt.show()

# Prepare data for SVM
# Features: Sugar and Flour
sugar_flour = recipes[['Sugar', 'Flour']].values
# Labels: 0 for Muffin, 1 for Cupcake
type_label = np.where(recipes['Type'] == 'Muffin', 0, 1)

# Initialize and fit the SVM model (Linear Kernel)
print("\n--- Training SVM Model (Linear Kernel) on all data ---")
model = svm.SVC(kernel='linear')
model.fit(sugar_flour, type_label)

# --- Plotting the SVM Decision Boundary ---
print("--- Plotting Decision Boundary and Margins ---")

# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]  # Slope of the hyperplane
xx = np.linspace(5, 30) # X-values for the line
# Y-values for the hyperplane
yy = a * xx - (model.intercept_[0]) / w[1]

# Get the margins (lines passing through support vectors)
# Margin passing through the first support vector
b_down = model.support_vectors_[0]
yy_down = a * xx + (b_down[1] - a * b_down[0])
# Margin passing through the last support vector
b_up = model.support_vectors_[-1]
yy_up = a * xx + (b_up[1] - a * b_up[0])

# Plot the data points
sns.lmplot(x='Sugar', y='Flour', data=recipes, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})

# Plot the hyperplane and margins
plt.plot(xx, yy, linewidth=2, color='black', label='Hyperplane')
plt.plot(xx, yy_down, 'k--', label='Margin (down)')
plt.plot(xx, yy_up, 'k--', label='Margin (up)')

# Highlight the support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none', edgecolors='k', label='Support Vectors')

plt.title('SVM Decision Boundary and Margins')
plt.legend()
plt.show()


# --- Train/Test Split and Evaluation ---
print("\n--- Evaluating Model on a Train/Test Split ---")
x_train, x_test, y_train, y_test = train_test_split(sugar_flour, type_label, test_size=0.2, random_state=42)

model1 = svm.SVC(kernel='linear')
model1.fit(x_train, y_train)

pred = model1.predict(x_test)
print(f"Predictions: {pred}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

