
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Iris Flower Analytics",
    page_icon="🌸",
    layout="wide"
)

# =====================================================
# CUSTOM DARK THEME
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #090D27, #171B46, #111A3D);
    color: white;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4, p, label {
    color: white !important;
}

div[data-testid="stMetric"] {
    background: #192951;
    border: 1px solid #6579A8;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: white;
}

div[data-testid="stForm"] {
    background: rgba(25, 41, 81, 0.35);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #293C68;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #E94B72, #9C4DCE);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    color: white;
    border: 1px solid white;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
}

hr {
    border-color: #293C68;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# CONSTANTS
# =====================================================

DATA_PATH = Path(__file__).resolve().parent / "Iris.csv"

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

TARGET = "Species"

# =====================================================
# LOAD DATASET
# =====================================================

if not DATA_PATH.exists():
    st.error("Iris.csv must be in the same folder as app.py.")
    st.stop()


@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    data.columns = data.columns.str.strip()

    # Validate required columns
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert measurements to numeric
    for col in FEATURES:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Clean species names
    data[TARGET] = (
        data[TARGET]
        .astype("string")
        .str.strip()
    )

    # Remove only missing or invalid required values
    data = data.dropna(
        subset=FEATURES + [TARGET]
    )

    data = data[
        data[TARGET] != ""
    ]

    # Preserve repeated valid observations
    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()

if df.empty:
    st.error("No valid records found in Iris.csv.")
    st.stop()

# =====================================================
# TRAIN MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def train_model(data):

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200)
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, y_test, predictions, accuracy


model, y_test, y_pred, accuracy = train_model(df)

# =====================================================
# PLOTLY DARK THEME HELPER
# =====================================================

def style_chart(fig):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#E5E7EB",
            family="Arial"
        ),
        title_font=dict(
            size=18,
            color="white"
        ),
        legend_title_font=dict(
            color="white"
        ),
        margin=dict(
            l=30,
            r=30,
            t=65,
            b=35
        )
    )

    fig.update_xaxes(
        gridcolor="rgba(180,190,220,0.15)"
    )

    fig.update_yaxes(
        gridcolor="rgba(180,190,220,0.15)"
    )

    return fig


# =====================================================
# DASHBOARD HEADER
# =====================================================

st.html("""
<div style="
background:linear-gradient(100deg,#401D43,#18274D);
padding:25px;
border-radius:14px;
border:1px solid #65729A;
">
<h1 style="color:white;margin:0;">
🌸 IRIS FLOWER CLASSIFICATION
</h1>
<p style="color:#C5D0E8;">
CODEALPHA DATA SCIENCE INTERNSHIP | TASK 1
</p>
<p style="color:#C5D0E8;">
Interactive Machine Learning and Dataset Analytics Dashboard
</p>
</div>
""")

# =====================================================
# KEY PERFORMANCE INDICATORS
# =====================================================

st.markdown("### 📊 KEY PERFORMANCE INDICATORS")

a, b, c, d = st.columns(4)

a.metric(
    "Dataset Records",
    len(df)
)

b.metric(
    "Input Features",
    len(FEATURES)
)

c.metric(
    "Flower Species",
    df[TARGET].nunique()
)

d.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.divider()

# =====================================================
# LIVE FLOWER PREDICTION
# =====================================================

st.subheader("🔮 LIVE FLOWER PREDICTION")

st.caption(
    "Enter flower measurements below to predict the Iris species."
)

with st.form("prediction_form"):

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        sl = st.number_input(
            "Sepal Length (cm)",
            min_value=0.0,
            max_value=15.0,
            value=5.1,
            step=0.1
        )

    with c2:
        sw = st.number_input(
            "Sepal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1
        )

    with c3:
        pl = st.number_input(
            "Petal Length (cm)",
            min_value=0.0,
            max_value=15.0,
            value=1.4,
            step=0.1
        )

    with c4:
        pw = st.number_input(
            "Petal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.1
        )

    submitted = st.form_submit_button(
        "🌸 Predict Species",
        width="stretch"
    )

if submitted:

    sample = pd.DataFrame(
        [[sl, sw, pl, pw]],
        columns=FEATURES
    )

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]

    confidence = max(probabilities) * 100

    st.markdown("### Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.success(
            f"🌸 Predicted Species: **{prediction}**"
        )

    with result_col2:
        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    probability_df = pd.DataFrame({
        "Species": model.classes_,
        "Probability": probabilities * 100
    })

    fig = px.bar(
        probability_df,
        x="Species",
        y="Probability",
        color="Species",
        text_auto=".2f",
        title="Prediction Probability by Species"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_yaxes(
        title="Probability (%)",
        range=[0, 110]
    )

    st.plotly_chart(
        style_chart(fig),
        width="stretch"
    )

# =====================================================
# DATASET ANALYSIS
# =====================================================

st.divider()

st.subheader("📊 DATASET ANALYSIS")

species = sorted(
    df[TARGET].unique()
)

selected = st.multiselect(
    "Select Flower Species",
    species,
    default=species
)

filtered = df[
    df[TARGET].isin(selected)
]

if filtered.empty:

    st.warning(
        "Please select at least one flower species."
    )

else:

    col1, col2 = st.columns(2)

    # Species Distribution
    with col1:

        fig1 = px.histogram(
            filtered,
            x=TARGET,
            color=TARGET,
            title="Species Distribution",
            text_auto=True
        )

        st.plotly_chart(
            style_chart(fig1),
            width="stretch"
        )

    # Petal Length vs Width
    with col2:

        fig2 = px.scatter(
            filtered,
            x="PetalLengthCm",
            y="PetalWidthCm",
            color=TARGET,
            title="Petal Length vs Width",
            hover_data=FEATURES
        )

        st.plotly_chart(
            style_chart(fig2),
            width="stretch"
        )

    col3, col4 = st.columns(2)

    # Petal Length Boxplot
    with col3:

        fig3 = px.box(
            filtered,
            x=TARGET,
            y="PetalLengthCm",
            color=TARGET,
            title="Petal Length by Species"
        )

        st.plotly_chart(
            style_chart(fig3),
            width="stretch"
        )

    # Feature Correlation
    with col4:

        corr = filtered[FEATURES].corr()

        fig4 = px.imshow(
            corr,
            text_auto=".2f",
            title="Feature Correlation",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )

        st.plotly_chart(
            style_chart(fig4),
            width="stretch"
        )

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.divider()

st.subheader("🎯 MODEL PERFORMANCE")

col1, col2 = st.columns(2)

# Confusion Matrix
with col1:

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    cm_df = pd.DataFrame(
        cm,
        index=model.classes_,
        columns=model.classes_
    )

    fig5 = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix",
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted Species",
            "y": "Actual Species",
            "color": "Count"
        }
    )

    st.plotly_chart(
        style_chart(fig5),
        width="stretch"
    )

# Classification Report
with col2:

    report = classification_report(
        y_test,
        y_pred,
        labels=model.classes_,
        target_names=model.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose().round(3)

    st.markdown("#### Classification Report")

    st.dataframe(
        report_df,
        width="stretch"
    )

# =====================================================
# DATASET TABLE AND DOWNLOAD
# =====================================================

st.divider()

st.subheader("📋 IRIS DATASET")

st.caption(
    f"Showing {len(filtered)} records from the selected species."
)

st.dataframe(
    filtered,
    width="stretch",
    hide_index=True
)

csv = filtered.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Dataset",
    data=csv,
    file_name="iris_dataset.csv",
    mime="text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.html("""
<div style="
background:#18254A;
border:1px solid #6579A8;
border-radius:12px;
padding:25px;
text-align:center;
">
<h3 style="color:white;">
🌸 IRIS FLOWER CLASSIFICATION
</h3>
<p style="color:#C5D0E8;">
Python | Pandas | Scikit-learn | Streamlit | Plotly
</p>
<p style="color:#AFC1E8;">
CodeAlpha Data Science Internship | Task 1
</p>
</div>
""")
