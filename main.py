
# ============================================================
# CODEALPHA DATA SCIENCE INTERNSHIP
# TASK 1: IRIS FLOWER CLASSIFICATION
# ============================================================

# -------------------- 1. IMPORT LIBRARIES --------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -------------------- 2. CREATE OUTPUT FOLDER --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n==========================================")
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("CODEALPHA DATA SCIENCE INTERNSHIP")
print("==========================================")


# -------------------- 3. LOAD DATASET --------------------

print("\nSTEP 1: LOADING DATASET")

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = pd.Categorical.from_codes(
    iris.target,
    iris.target_names
)

print("Dataset loaded successfully!")

print("\nFirst Five Records:")
print(df.head())


# -------------------- 4. DATA EXPLORATION --------------------

print("\nSTEP 2: DATA EXPLORATION")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nSpecies Distribution:")
print(df["species"].value_counts())

print("\nTotal Records:", df.shape[0])
print("Total Features:", len(iris.feature_names))
print("Total Species:", df["species"].nunique())


# -------------------- 5. VISUALIZATION 1 --------------------
# Species Distribution

print("\nSTEP 3: CREATING VISUALIZATIONS")

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="species"
)

plt.title("Iris Flower Species Distribution")
plt.xlabel("Flower Species")
plt.ylabel("Number of Flowers")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "species_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -------------------- 6. VISUALIZATION 2 --------------------
# Pairplot

pair_plot = sns.pairplot(
    df,
    hue="species"
)

pair_plot.fig.suptitle(
    "Iris Flower Feature Relationships",
    y=1.02
)

pair_plot.savefig(
    os.path.join(OUTPUT_DIR, "pairplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close("all")


# -------------------- 7. VISUALIZATION 3 --------------------
# Correlation Heatmap

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Iris Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -------------------- 8. VISUALIZATION 4 --------------------
# Petal Length Boxplot

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="species",
    y="petal length (cm)"
)

plt.title("Petal Length by Iris Species")
plt.xlabel("Flower Species")
plt.ylabel("Petal Length (cm)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "petal_boxplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -------------------- 9. DATA PREPROCESSING --------------------

print("\nSTEP 4: DATA PREPROCESSING")

# Input features
X = iris.data

# Target labels
y = iris.target

print("Input Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# -------------------- 10. MODEL TRAINING --------------------

print("\nSTEP 5: MACHINE LEARNING MODEL TRAINING")

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200)
)

# Train the model
model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")


# -------------------- 11. MODEL PREDICTION --------------------

print("\nSTEP 6: MODEL PREDICTION")

y_pred = model.predict(X_test)

print("\nPredicted Species:")
print(iris.target_names[y_pred])

print("\nActual Species:")
print(iris.target_names[y_test])


# -------------------- 12. MODEL EVALUATION --------------------

print("\nSTEP 7: MODEL EVALUATION")

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:",
      round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# -------------------- 13. CONFUSION MATRIX --------------------

print("\nSTEP 8: CONFUSION MATRIX")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Iris Flower Classification Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -------------------- 14. NEW FLOWER PREDICTION --------------------

print("\nSTEP 9: TESTING NEW FLOWER")

# Sepal length, Sepal width, Petal length, Petal width
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

predicted_species = iris.target_names[prediction[0]]

print("\nNew Flower Measurements:", sample)

print("Predicted Flower Species:", predicted_species)


# -------------------- 15. SAVE RESULTS --------------------

print("\nSTEP 10: SAVED VISUALIZATION FILES")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("Saved:", filename)

print("\nImage Folder:", OUTPUT_DIR)


# -------------------- 16. FINAL MESSAGE --------------------

print("\n==========================================")
print("IRIS FLOWER CLASSIFICATION COMPLETED!")
print("==========================================")

print("Dataset Records:", len(df))
print("Number of Species:", df["species"].nunique())
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("All output graphs saved successfully!")
print("==========================================")
