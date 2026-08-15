import streamlit as st
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.lib.libstreamlit import grid_search_rf, train_data_processing, test_data_processing


PRESENTATION_PDF_PATH = "SECOM Final Presentation.pdf"

# Load the SECOM dataset
X = pd.read_csv("data/secom.data", sep=" ", header=None, na_values="NaN")
y = pd.read_csv("data/secom_labels.data", sep=" ", header=None, names=["label", "timestamp"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y["label"],
    test_size=0.2,
    random_state=42,
    stratify=y["label"],
)
print("Original:", X.shape, "| Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# Map labels {-1, 1} -> {0, 1}
y_train = y_train.replace({-1: 0})
y_test = y_test.replace({-1: 0})


def run_experiment(
    outlier_handling,
    missingness_threshold,
    scaling_method,
    knn_params,
    balancing_handling,
):
    """
    Run one full experiment using the current SECOM train/test split
    and the user-selected preprocessing decisions.
    """

    # 1. Training data processing
    X_train_balanced, y_train_balanced, X_train_reduced, selected_features, imputer, scaler = train_data_processing(
        X_train,
        y_train,
        outlier_handling=outlier_handling,
        missingness_threshold=missingness_threshold,
        scaling_method=scaling_method,
        knn_imputation=knn_params,
        balancing_handling=balancing_handling
    )

    # 2. Test data processing (using training pipeline objects)
    X_test_selected = test_data_processing(
        X_test,
        X_train_reduced,
        selected_features,
        imputer,
        scaler
    )

    # 3. Model training (RF grid search)
    best_model = grid_search_rf(X_train_balanced, y_train_balanced)

    # 4. Predictions
    y_pred = best_model.predict(X_test_selected)

    # 5. Metrics (only the 4 you want)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    model_info = {
        "model_name": type(best_model).__name__,
        "params": {
            "outlier_handling": outlier_handling,
            "missingness_threshold": missingness_threshold,
            "scaling_method": scaling_method,
            "knn_imputation": knn_params,
            "balancing_handling": balancing_handling
        },
    }

    return metrics, model_info


# --- Global store for experiments (in-memory for now) ---
if "experiments" not in st.session_state:
    st.session_state["experiments"] = []  # list of dicts


def main():
    st.set_page_config(page_title="Preprocessing & Hyperparameter Explorer", layout="wide")

    tab_run, tab_best = st.tabs(["Run Experiment", "Best Model & Presentation"])

    # =========================
    # Tab 1: Run Experiment
    # =========================
    with tab_run:
        st.header("Run a New Experiment")

        st.subheader("Dataset Preview")
        st.dataframe(X.head())

        st.markdown("### Choose preprocessing & modeling decisions")

        # --- Options from your scope ---
        outlier_options = ["cap", "remove"]
        missingness_threshold_options = [0.5, 0.6, 0.7, 0.8, 0.9]
        scaling_options = ["standard", "minmax", "robust"]
        knn_options = [
            {"n_neighbors": 5, "weights": "uniform"},
            {"n_neighbors": 7, "weights": "uniform"},
            {"n_neighbors": 10, "weights": "uniform"},
            {"n_neighbors": 5, "weights": "distance"},
            {"n_neighbors": 7, "weights": "distance"},
            {"n_neighbors": 10, "weights": "distance"},
        ]
        balancing_options = ["smote", "adasyn", "random_oversample"]

        col1, col2 = st.columns(2)

        with col1:
            outlier_choice = st.selectbox("Outlier handling", outlier_options)
            missingness_choice = st.selectbox("Missingness threshold", missingness_threshold_options)
            scaling_choice = st.selectbox("Scaling method", scaling_options)

        with col2:
            knn_labels = [
                f"n_neighbors={opt['n_neighbors']}, weights={opt['weights']}"
                for opt in knn_options
            ]
            knn_index = st.selectbox(
                "KNN imputation",
                range(len(knn_options)),
                format_func=lambda i: knn_labels[i],
            )
            knn_choice = knn_options[knn_index]

            balancing_choice = st.selectbox("Balancing handling", balancing_options)

        if st.button("Run training & evaluation"):
            with st.spinner("Running experiment..."):
                metrics, model_info = run_experiment(
                    outlier_handling=outlier_choice,
                    missingness_threshold=missingness_choice,
                    scaling_method=scaling_choice,
                    knn_params=knn_choice,
                    balancing_handling=balancing_choice,
                )

            st.success("Experiment completed!")

            st.markdown("### Evaluation metrics")
            st.write(f"Accuracy: **{metrics['accuracy']:.4f}**")
            st.write(f"Precision: **{metrics['precision']:.4f}**")
            st.write(f"Recall: **{metrics['recall']:.4f}**")
            st.write(f"F1 Score: **{metrics['f1']:.4f}**")

            st.markdown("### Model configuration")
            st.json(model_info["params"])

            # Save experiment to session_state
            st.session_state["experiments"].append(
                {
                    "metrics": metrics,
                    "model_info": model_info,
                }
            )

    # =========================
    # Tab 2: Best Model & PDF
    # =========================
    with tab_best:
        st.header("Best Model Summary & Presentation")

        st.markdown("### Best Pipeline (Static)")
        best_pipeline = {
            "Outlier Handling": "cap",
            "Missingness Threshold": 0.6,
            "Scaling Method": "robust",
            "Imputation": {"n_neighbors": 5, "weights": "uniform"},
            "Balancing Algorithm": "smote",
        }
        st.json(best_pipeline)

        st.markdown("### Best Model Metrics (Static)")
        best_metrics = {
            "accuracy": 0.58,
            "precision": 0.11,
            "recall": 0.71,
            "f1": 0.18,
            "auc": 0.684,
        }

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Accuracy: **{best_metrics['accuracy']:.2f}**")
            st.write(f"Precision: **{best_metrics['precision']:.2f}**")
            st.write(f"Recall: **{best_metrics['recall']:.2f}**")
            st.write(f"F1 Score: **{best_metrics['f1']:.2f}**")
            st.write(f"AUC: **{best_metrics['auc']:.3f}**")

        st.markdown("### Confusion Matrix (Static)")
        cm = {
            "True Negative": 166,
            "False Positive": 127,
            "False Negative": 6,
            "True Positive": 15,
        }
        st.json(cm)

        # st.markdown("### ROC Curve (Static)")
        # st.image("reports/roc_curve.png", caption="ROC Curve (AUC = 0.684)")

        st.markdown("### Project Presentation (PDF)")
        try:
            with open("SECOM Final Presentation.pdf", "rb") as f:
                st.download_button(
                label="Download PDF",
                data=f,
                file_name="SECOM Final Presentation.pdf",
                mime="application/pdf"
            )
        except FileNotFoundError:
            st.warning("Presentation PDF not found. Check PRESENTATION_PDF_PATH.")
        st.pdf("SECOM Final Presentation.pdf")
if __name__ == "__main__":
    main()
