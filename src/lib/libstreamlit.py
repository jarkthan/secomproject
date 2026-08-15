import numpy as np
import pandas as pd
from src.lib.dataprep import Cleaning, Imputation, RoughFeatureReduction, FeatureSelector, Balancer
from src.lib.datamodeling import DataModeler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# This function processes the training data based on the specified parameters:
# outlier handling, missingness threshold, scaling method, KNN imputation, and balancing handling. 
# It returns the processed training data along with the selected features, imputer, and scaler.
def train_data_processing(
        X_train, y_train, 
        outlier_handling,
        missingness_threshold,
        scaling_method,
        knn_imputation,
        balancing_handling
        # param_grid={}
) -> dict:
    # Data Preprocessing
    X_train_cleaned, _ = Cleaning().flag_missing_values(X_train)
    
    # Outlier handling
    if outlier_handling == 'cap':
        X_train_cleaned = Cleaning().cap_outlier_using_3sigma_rule(X_train_cleaned)
    elif outlier_handling == 'remove':
        X_train_cleaned = Cleaning().mark_outlier_3sigma_as_nan(X_train_cleaned)

    # Rough feature reduction
    X_train_reduced = RoughFeatureReduction().remove_constant_features(X_train_cleaned)
    X_train_reduced = RoughFeatureReduction().remove_features_by_missingness_threshold(X_train_reduced, threshold=missingness_threshold)

    # Impute and Scale missing values
    imputer = Imputation().fit_imputer(X_train_reduced, n_neighbors=knn_imputation['n_neighbors'], weights=knn_imputation['weights'])
    X_train_imputed = Imputation().transform_imputer(X_train_reduced, imputer)
    scaler = Imputation().fit_scaler(X_train_imputed, scaling_method=scaling_method)
    x_train_scaled = Imputation().transform_scaler(X_train_imputed, scaler)


    # Feature selection
    X_train_selected, selected_features = FeatureSelector().boruta_select(x_train_scaled, y_train)
    
    # Feature balancing
    if balancing_handling=='smote':
        X_train_balanced, y_train_balanced = Balancer().apply_smote(X_train_selected, y_train)

    elif balancing_handling=='adasyn':
        X_train_balanced, y_train_balanced = Balancer().apply_adasyn(X_train_selected, y_train)

    elif balancing_handling=='random_oversample':
        X_train_balanced, y_train_balanced = Balancer().apply_random_oversample(X_train_selected, y_train)

    else:
        error_message = f"Invalid balancing handling method: {balancing_handling}. Choose from 'smote', 'adasyn', or 'random_oversample'."
        raise ValueError(error_message)
    
    return X_train_balanced, y_train_balanced, X_train_reduced, selected_features, imputer, scaler

# This function processes the test data based on the selected features, imputer, and scaler from the training data.
def test_data_processing(X_test, X_train_reduced, selected_features, imputer, scaler):
    X_test_selected = X_test[X_train_reduced.columns]
    X_test_selected = imputer.transform(X_test_selected)
    X_test_selected = scaler.transform(X_test_selected)
    X_test_selected = pd.DataFrame(X_test_selected, columns=X_train_reduced.columns)
    X_test_selected = X_test_selected[selected_features]

    # print(f"{ts()} Test data processed. Shape: {X_test_selected.shape}")
    return X_test_selected

# This function performs a grid search to find the best hyperparameters for a Random Forest classifier.
def grid_search_rf(X_train_balanced, y_train_balanced):
    param_grid = {
    "n_estimators": [200, 300, 400, 500, 600],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_leaf": [1, 3, 5 ,7],
    "class_weight": ["balanced", {0:1, 1:3}, {0:1, 1:5}]
}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        scoring="recall",          # focus on detecting faulty wavers
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_balanced, y_train_balanced.values.ravel())

    print("Best params:", grid.best_params_)
    # print("Best CV recall:", grid.best_score_)
    best_model = grid.best_estimator_
    return best_model