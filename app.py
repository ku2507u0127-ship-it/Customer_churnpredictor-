from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics_summary.json"
FEATURES_PATH = BASE_DIR / "artifacts" / "feature_importance.csv"


def build_fallback_pipeline(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=["Target"])
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X, df["Target"])
    return pipeline


@st.cache_data
def load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATA_PATH)
    if dataset["Target"].dtype == "object":
        dataset["Target"] = dataset["Target"].map({"Yes": 1, "No": 0})
    return dataset


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    if FEATURES_PATH.exists():
        return pd.read_csv(FEATURES_PATH)
    return pd.DataFrame(columns=["feature", "importance"])


@st.cache_resource
def load_model() -> Pipeline:
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return build_fallback_pipeline(load_dataset())


@st.cache_data
def compute_live_metrics(dataset: pd.DataFrame) -> dict:
    X = dataset.drop(columns=["Target"])
    y = dataset["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_fallback_pipeline(pd.concat([X_train, y_train], axis=1))
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "dataset_shape": {"rows": int(dataset.shape[0]), "columns": int(dataset.shape[1])},
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def prediction_label(probability: float) -> str:
    return "High churn risk" if probability >= 0.5 else "Low churn risk"


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn Prediction Dashboard")
st.caption("Random Forest based churn classification with an interactive prediction form.")

metrics = load_metrics()
feature_importance = load_feature_importance()
dataset = load_dataset()
model = load_model()
if not metrics:
    metrics = compute_live_metrics(dataset)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Rows", metrics.get("dataset_shape", {}).get("rows", len(dataset)))
metric_col2.metric("Model Accuracy", f"{metrics.get('accuracy', 0):.2%}")
metric_col3.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
metric_col4.metric("Churn Rate", f"{dataset['Target'].mean():.2%}")

st.divider()

left_col, right_col = st.columns([1.15, 0.85])

with left_col:
    st.subheader("Predict Customer Churn")
    with st.form("prediction_form"):
        age = st.slider("Age", min_value=int(dataset["Age"].min()), max_value=int(dataset["Age"].max()), value=34)
        services_opted = st.slider(
            "Services Opted",
            min_value=int(dataset["ServicesOpted"].min()),
            max_value=int(dataset["ServicesOpted"].max()),
            value=3,
        )
        frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes"])
        annual_income = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])
        social_sync = st.selectbox("Account Synced To Social Media", ["No", "Yes"])
        booked_hotel = st.selectbox("Booked Hotel Or Not", ["No", "Yes"])
        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        sample = pd.DataFrame(
            [
                {
                    "Age": age,
                    "FrequentFlyer": frequent_flyer,
                    "AnnualIncomeClass": annual_income,
                    "ServicesOpted": services_opted,
                    "AccountSyncedToSocialMedia": social_sync,
                    "BookedHotelOrNot": booked_hotel,
                }
            ]
        )
        probability = float(model.predict_proba(sample)[0][1])
        st.success(f"Prediction: {prediction_label(probability)}")
        st.progress(min(max(probability, 0.0), 1.0), text=f"Predicted churn probability: {probability:.2%}")

with right_col:
    st.subheader("Project Snapshot")
    st.write(
        """
        This application uses a Random Forest model trained on customer demographics,
        account signals, and service usage behavior. It helps estimate whether a customer
        is likely to churn so the business can take preventive retention actions.
        """
    )

    if not feature_importance.empty:
        chart_df = feature_importance.head(10).sort_values("importance")
        st.bar_chart(chart_df.set_index("feature"))

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Target Distribution")
    churn_counts = dataset["Target"].map({0: "No Churn", 1: "Churn"}).value_counts()
    st.bar_chart(churn_counts)

with chart_col2:
    st.subheader("Frequent Flyer vs Churn")
    flyer_pivot = pd.crosstab(dataset["FrequentFlyer"], dataset["Target"])
    flyer_pivot.columns = ["No Churn", "Churn"]
    st.bar_chart(flyer_pivot)

st.subheader("Dataset Preview")
st.dataframe(dataset.head(10), width="stretch")
