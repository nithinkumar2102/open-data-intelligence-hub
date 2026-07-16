from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page settings
# --------------------------------------------------
st.set_page_config(
    page_title="Optimized Classification Model",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# Correct project folder paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "customer_purchase_data.csv"
MODEL_FILE = BASE_DIR / "outputs" / "best_purchase_model.pkl"
COMPARISON_FILE = BASE_DIR / "outputs" / "model_comparison.csv"
IMPORTANCE_FILE = BASE_DIR / "outputs" / "feature_importance.csv"
CONFUSION_IMAGE = BASE_DIR / "outputs" / "confusion_matrix.png"
FEATURE_IMAGE = BASE_DIR / "outputs" / "feature_importance.png"
REPORT_FILE = BASE_DIR / "outputs" / "classification_report.txt"


# --------------------------------------------------
# Main title
# --------------------------------------------------
st.title(
    "Optimized Classification Model with Feature Importance Analysis"
)

st.write(
    "This dashboard compares classification algorithms, displays model "
    "performance and predicts whether a customer will make a purchase."
)


# --------------------------------------------------
# Check all required files
# --------------------------------------------------
missing_files = []

required_files = [
    DATA_FILE,
    MODEL_FILE,
    COMPARISON_FILE,
    IMPORTANCE_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        missing_files.append(file_path.name)

if missing_files:
    st.error(
        "The following files are missing: "
        + ", ".join(missing_files)
    )

    st.info(
        "Run these commands in the terminal:\n\n"
        "python generate_dataset.py\n\n"
        "python model.py"
    )

    st.stop()


# --------------------------------------------------
# Load project files
# --------------------------------------------------
try:
    dataset = pd.read_csv(DATA_FILE)

    model_comparison = pd.read_csv(
        COMPARISON_FILE
    )

    feature_importance = pd.read_csv(
        IMPORTANCE_FILE
    )

    trained_model = joblib.load(
        MODEL_FILE
    )

except Exception as error:
    st.error(
        f"Unable to load project files: {error}"
    )
    st.stop()


# --------------------------------------------------
# Dataset overview
# --------------------------------------------------
st.header("Dataset Overview")

metric1, metric2, metric3, metric4 = st.columns(4)

total_customers = len(dataset)

purchased_customers = int(
    dataset["Purchased"].sum()
)

not_purchased_customers = int(
    (dataset["Purchased"] == 0).sum()
)

total_features = len(
    dataset.drop(
        columns=["Purchased"]
    ).columns
)

metric1.metric(
    "Total Customers",
    total_customers,
)

metric2.metric(
    "Purchased",
    purchased_customers,
)

metric3.metric(
    "Not Purchased",
    not_purchased_customers,
)

metric4.metric(
    "Total Features",
    total_features,
)


# --------------------------------------------------
# Dataset preview
# --------------------------------------------------
st.subheader("Dataset Preview")

st.dataframe(
    dataset.head(20),
    use_container_width=True,
)


# --------------------------------------------------
# Purchase distribution
# --------------------------------------------------
st.subheader("Purchase Distribution")

purchase_distribution = (
    dataset["Purchased"]
    .value_counts()
    .sort_index()
)

purchase_distribution.index = [
    "Not Purchased",
    "Purchased",
]

st.bar_chart(
    purchase_distribution
)


# --------------------------------------------------
# Model comparison
# --------------------------------------------------
st.header("Model Comparison")

st.dataframe(
    model_comparison,
    use_container_width=True,
)

if (
    "Model" in model_comparison.columns
    and "Accuracy" in model_comparison.columns
):
    comparison_chart = (
        model_comparison
        .set_index("Model")
    )

    st.bar_chart(
        comparison_chart["Accuracy"]
    )

    best_row = model_comparison.sort_values(
        by="Accuracy",
        ascending=False,
    ).iloc[0]

    best_model_name = best_row["Model"]
    best_accuracy = best_row["Accuracy"]

    st.success(
        f"Best Model: {best_model_name}"
    )

    st.info(
        f"Best Accuracy: {best_accuracy:.2%}"
    )


# --------------------------------------------------
# Feature importance
# --------------------------------------------------
st.header("Feature Importance Analysis")

st.dataframe(
    feature_importance,
    use_container_width=True,
)

if (
    "Feature" in feature_importance.columns
    and "Importance" in feature_importance.columns
):
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.barh(
        feature_importance["Feature"],
        feature_importance["Importance"],
    )

    axis.set_title(
        "Feature Importance"
    )

    axis.set_xlabel(
        "Importance Score"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.invert_yaxis()

    plt.tight_layout()

    st.pyplot(figure)

    plt.close(figure)


# --------------------------------------------------
# Model evaluation images
# --------------------------------------------------
st.header("Model Evaluation")

image_column1, image_column2 = st.columns(2)

with image_column1:
    st.subheader("Confusion Matrix")

    if CONFUSION_IMAGE.exists():
        st.image(
            str(CONFUSION_IMAGE),
            use_container_width=True,
        )
    else:
        st.warning(
            "Confusion matrix image is not available."
        )

with image_column2:
    st.subheader("Saved Feature Importance")

    if FEATURE_IMAGE.exists():
        st.image(
            str(FEATURE_IMAGE),
            use_container_width=True,
        )
    else:
        st.warning(
            "Feature importance image is not available."
        )


# --------------------------------------------------
# Classification report
# --------------------------------------------------
st.header("Classification Report")

if REPORT_FILE.exists():
    try:
        report_text = REPORT_FILE.read_text(
            encoding="utf-8"
        )

        st.code(
            report_text,
            language="text",
        )

    except Exception as error:
        st.warning(
            f"Unable to read classification report: {error}"
        )

else:
    st.warning(
        "Classification report file is not available."
    )


# --------------------------------------------------
# Prediction form
# --------------------------------------------------
st.header("Customer Purchase Prediction")

st.write(
    "Enter customer details to predict whether "
    "the customer will make a purchase."
)

input_column1, input_column2 = st.columns(2)

with input_column1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
    )

    annual_income = st.number_input(
        "Annual Income",
        min_value=10000,
        max_value=500000,
        value=60000,
        step=1000,
    )

    purchase_frequency = st.number_input(
        "Purchase Frequency",
        min_value=1,
        max_value=100,
        value=12,
        step=1,
    )

with input_column2:
    average_order_value = st.number_input(
        "Average Order Value",
        min_value=100,
        max_value=100000,
        value=3000,
        step=100,
    )

    website_visits = st.number_input(
        "Website Visits",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
    )

    discount_used = st.selectbox(
        "Discount Used",
        options=[0, 1],
        format_func=lambda value: (
            "Yes" if value == 1 else "No"
        ),
    )


# --------------------------------------------------
# Prediction button
# --------------------------------------------------
if st.button(
    "Predict Purchase",
    type="primary",
):
    customer_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Annual_Income": annual_income,
                "Purchase_Frequency": purchase_frequency,
                "Average_Order_Value": average_order_value,
                "Website_Visits": website_visits,
                "Discount_Used": discount_used,
            }
        ]
    )

    try:
        prediction = trained_model.predict(
            customer_data
        )[0]

        if prediction == 1:
            st.success(
                "The customer is likely to make a purchase."
            )

        else:
            st.warning(
                "The customer is not likely to make a purchase."
            )

        if hasattr(
            trained_model,
            "predict_proba",
        ):
            probabilities = trained_model.predict_proba(
                customer_data
            )[0]

            purchase_probability = probabilities[1]

            st.info(
                f"Purchase Probability: "
                f"{purchase_probability:.2%}"
            )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Task 9 - Optimized Classification Model "
    "with Feature Importance Analysis"
)