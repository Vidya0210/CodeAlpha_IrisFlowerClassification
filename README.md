# 🌸 Iris Flower Classification Using Machine Learning

## 📌 Project Overview

This project is developed as part of the **CodeAlpha Data Science Internship – Task 1**.

The objective is to build a Machine Learning classification model that predicts the species of an Iris flower based on its physical measurements.

The model classifies flowers into three species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

## 🎯 Project Objectives

- Understand and explore the Iris dataset.
- Perform Exploratory Data Analysis (EDA).
- Visualize relationships between flower measurements.
- Preprocess data for Machine Learning.
- Train a classification model.
- Evaluate model performance using accuracy and a confusion matrix.
- Predict the species of a new flower.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## 📊 Dataset Description

The project uses the built-in Iris dataset available in Scikit-learn.

**Dataset information:**
- Total records: 150
- Input features: 4
- Target classes: 3
- Missing values: 0

**Input Features:**
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

**Target:** Flower species.

## 🔍 Exploratory Data Analysis

The following visualizations were created:

1. Species Distribution
2. Pairplot – Feature Relationships
3. Correlation Heatmap
4. Petal Length Boxplot

## 🤖 Machine Learning Model

**Algorithm:** Logistic Regression

**Preprocessing:**
- Separated input features and target labels.
- Split the dataset into training and testing sets.
- Applied StandardScaler for feature scaling.

**Dataset Split:**
- Training data: 80% (120 records)
- Testing data: 20% (30 records)

## 📈 Model Performance

**Achieved Accuracy: 93.33%**

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 📁 Project Structure

```text
CodeAlpha_IrisFlowerClassification/
│
├── main.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── species_distribution.png
    ├── pairplot.png
    ├── correlation_heatmap.png
    ├── petal_boxplot.png
    └── confusion_matrix.png
```

## ⚙️ Installation and Execution

**Step 1: Clone the repository**

```bash
git clone https://github.com/Vidya0210/CodeAlpha_IrisFlowerClassification.git
```

**Step 2: Navigate to the project folder**

```bash
cd CodeAlpha_IrisFlowerClassification
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Run the project**

```bash
python main.py
```

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Data loading and preprocessing.
- Exploratory Data Analysis.
- Data visualization.
- Classification algorithms.
- Model training and testing.
- Machine Learning model evaluation.

## 👩‍💻 Internship Information

**Organization:** CodeAlpha

**Domain:** Data Science

**Task:** Task 1 – Iris Flower Classification

**GitHub Repository:** https://github.com/Vidya0210/CodeAlpha_IrisFlowerClassification

---

*Developed as part of the CodeAlpha Data Science Internship.*
