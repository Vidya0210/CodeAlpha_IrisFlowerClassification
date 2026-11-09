# 🌸 Iris Flower Classification – Interactive Analytics Dashboard

## 📌 Project Overview

This project is developed as part of the **CodeAlpha Data Science Internship – Task 1**.

It uses Machine Learning to classify Iris flowers into three species based on four flower measurements. An interactive Streamlit dashboard allows users to enter measurements, predict flower species, explore the dataset, and evaluate model performance.

## 🎯 Objectives

- Explore and analyze the Iris flower dataset.
- Build a Machine Learning classification model.
- Predict flower species using user-provided measurements.
- Visualize relationships between flower features.
- Evaluate model performance using classification metrics.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn

## 📊 Dataset Information

The project uses `Iris.csv`, containing 150 flower observations and the following columns:

| Feature       | Description        |
| ------------- | ------------------ |
| SepalLengthCm | Sepal length in cm |
| SepalWidthCm  | Sepal width in cm  |
| PetalLengthCm | Petal length in cm |
| PetalWidthCm  | Petal width in cm  |
| Species       | Flower species     |

The three flower species are:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The `Id` column is not used as a Machine Learning feature.

## 🤖 Machine Learning Model

**Algorithm:** Logistic Regression

**Preprocessing:** StandardScaler

**Train-test split:** 80% training and 20% testing

**Random state:** 42

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

**Observed test accuracy:** 93.33%

## ✨ Dashboard Features

1. **KPI Cards:** Dataset records, input features, species count, and model accuracy.
2. **Live Flower Prediction:** Predict flower species using four input measurements.
3. **Prediction Confidence:** Display estimated prediction probabilities.
4. **Species Distribution:** Visualize the distribution of flower species.
5. **Feature Relationship:** Explore petal length versus petal width.
6. **Box Plot:** Compare petal lengths across flower species.
7. **Correlation Heatmap:** Analyze relationships between numerical features.
8. **Model Evaluation:** View confusion matrix and classification report.
9. **Dataset Viewer:** Explore and download filtered dataset records.

## 📁 Project Structure

```text
CodeAlpha_IrisFlowerClassification/
│
├── Iris.csv
├── app.py
├── main.py
├── check_dataset.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── confusion_matrix.png
    ├── correlation_heatmap.png
    ├── pairplot.png
    ├── petal_boxplot.png
    └── species_distribution.png
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vidya0210/CodeAlpha_IrisFlowerClassification.git
```

### 2. Navigate to the project folder

```bash
cd CodeAlpha_IrisFlowerClassification
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
python -m streamlit run app.py
```

### 5. Open the application

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

## 📈 Results

- Successfully loaded 150 valid dataset observations.
- Classified three Iris flower species.
- Achieved 93.33% test accuracy in the displayed evaluation.
- Built an interactive dashboard with live prediction and visual analytics.

## 👩‍💻 Internship Information

**Program:** CodeAlpha Data Science Internship

**Task:** Task 1 – Iris Flower Classification

## 🔗 GitHub Repository

[CodeAlpha Iris Flower Classification](https://github.com/Vidya0210/CodeAlpha_IrisFlowerClassification)

---

_Developed as part of my journey in Data Science and Machine Learning._ 🌸