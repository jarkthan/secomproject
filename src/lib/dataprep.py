import numpy as np
import pandas as pd
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class Cleaning:
    def __init__(self) -> None:
        pass

    def flag_missing_values(self, data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """
        Description:
            Replace string values that may indicate missing data with NaN.
        Parameters:
            df (pd.DataFrame): The input DataFrame to clean.
        Returns:
            tuple: A tuple containing the modified DataFrame and a boolean indicating if any string values are still found.
        """
        df = data.replace(["", " ", "?", "NA", "N/A", "nan", "NaN"], np.nan)
        string_mask = df.map(lambda v: isinstance(v, str))
        has_invalid_strings = string_mask.any().any()
        return df, has_invalid_strings

    def cap_outlier_using_3sigma_rule(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            Caps outliers in the DataFrame using the 3-sigma rule.
        Parameters:
            data (pd.DataFrame): The input DataFrame to clean.
        Returns:
            pd.DataFrame: The cleaned DataFrame with outliers capped.
        """
        means = data.mean()
        stds = data.std()
        lower_bound = means - 3 * stds
        upper_bound = means + 3 * stds
        for col in data.columns:
            data[col] = np.clip(
                data[col],
                lower_bound[col],
                upper_bound[col]
            )
        return data
    
class RoughFeatureReduction:
    def __init__(self) -> None:
        pass

    def remove_features_by_missingness_threshold(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Description:
            Remove features that have a percentage of missing values above the specified threshold.
        Parameters:
            data (pd.DataFrame): The input DataFrame to clean.
            threshold (float): The missingness threshold (between 0 and 1).
        Returns:
            pd.DataFrame: The cleaned DataFrame with features removed based on missingness threshold.
        """
        missing_percent = data.isnull().mean()
        features_to_remove = missing_percent[missing_percent > threshold].index.tolist()
        data.drop(columns=features_to_remove, inplace=True)
        return data

    def remove_constant_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            Remove features that have the same value across all samples.
        Parameters:
            data (pd.DataFrame): The input DataFrame to clean.
        Returns:
            pd.DataFrame: The cleaned DataFrame with constant features removed.
        """
        nunique = data.nunique()
        constant_features = nunique[nunique == 1].index.tolist()
        data.drop(columns=constant_features, inplace=True)
        return data
    
class ImputationOld:
    def __init__(self) -> None:
        pass

    def knn_scale_and_impute(self, data: pd.DataFrame, n_neighbors: int = 5) -> tuple[pd.DataFrame, StandardScaler, KNNImputer]:
        """
        Description:
            Impute missing values using K-Nearest Neighbors (KNN) imputation.
        Parameters:
            data (pd.DataFrame): The input DataFrame to clean.
            n_neighbors (int): The number of neighboring samples to use for imputation.
        Returns:
            tuple[pd.DataFrame, StandardScaler, KNNImputer]: The cleaned DataFrame with missing values imputed, along with the fitted scaler and imputer.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(scaled_data)
        return pd.DataFrame(imputed_data, columns=data.columns), scaler, imputer
    
class Imputation:
    def __init__(self):
        pass

    def fit_scaler(self, X: pd.DataFrame) -> StandardScaler:
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler

    def transform_scaler(self, X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        scaled = scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns)

    def fit_imputer(self, X: pd.DataFrame, n_neighbors=5) -> KNNImputer:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputer.fit(X)
        return imputer

    def transform_imputer(self, X: pd.DataFrame, imputer: KNNImputer) -> pd.DataFrame:
        imputed = imputer.transform(X)
        return pd.DataFrame(imputed, columns=X.columns)


class FeatureSelector:
    def __init__(self) -> None:
        pass

    def boruta_select(self, X: pd.DataFrame, y: pd.Series, max_iter: int = 100) -> tuple[pd.DataFrame, list]:
        """
        Run Boruta feature selection and return a reduced dataframe and the selected features.
        """

        # 1. Random Forest model for Boruta
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42
        )

        # 2. Initialize Boruta
        selector = BorutaPy(
            estimator=rf,
            n_estimators="auto",
            max_iter=max_iter,
            verbose=0,
            random_state=42
        )

        # 3. Fit Boruta
        selector.fit(X.values, y.values)

        # 4. Extract selected + tentative features
        selected = X.columns[selector.support_].tolist()
        tentative = X.columns[selector.support_weak_].tolist()

        # 5. Combine selected + tentative
        final_features = selected + tentative

        # 6. Return reduced dataframe
        return X[final_features], final_features
    
class Balancer:
    def __init__(self) -> None:
        pass

    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance the dataset.
        Returns balanced X and y.
        """

        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)

        print("Before SMOTE:")
        print(y.value_counts())
        print("\nAfter SMOTE:")
        print(y_balanced.value_counts())

        return X_balanced, y_balanced