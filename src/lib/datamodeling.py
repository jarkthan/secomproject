import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

class DataModeler:
    def __init__(self) -> None:
        pass

    def evaluate_model(self, y_test: pd.DataFrame, y_pred: pd.Series, evaluation_report: bool = True, return_cm_cost: bool = False) -> str:
        if evaluation_report:
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("\nClassification Report:\n", classification_report(y_test, y_pred))
            print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")

        if return_cm_cost:
            COST_FP = 100   # Type I
            COST_FN = 400   # Type II
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            total_cost = COST_FP * fp + COST_FN * fn
            return (tn, fp, fn, tp), total_cost
        
    def auc_and_roc_curve(self, y_test: pd.DataFrame, y_proba: pd.Series) -> None:
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()