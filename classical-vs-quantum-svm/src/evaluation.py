import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def evaluate_predictions(y_true, y_pred, train_time, predict_time):
    """
    Evaluate classification predictions and return metrics.
    """

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Debug information
    print("\n========== Evaluation ==========")
    print("Unique y_true :", np.unique(y_true))
    print("Unique y_pred :", np.unique(y_pred))
    print("First 10 y_true:", y_true[:10])
    print("First 10 y_pred:", y_pred[:10])

    unique_classes = np.unique(y_true)
    is_binary = len(unique_classes) == 2
    avg_strategy = 'binary' if is_binary else 'weighted'

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average=avg_strategy,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average=avg_strategy,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=avg_strategy,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("===============================\n")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "train_time": float(train_time),
        "predict_time": float(predict_time),
    }

def save_results_to_csv(results, csv_path):
    """
    Save evaluation results to a CSV file.
    """
    rows = []
    for (dataset, model), metrics in results.items():
        rows.append({
            "Dataset": dataset,
            "Model": model,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1_Score": metrics["f1_score"],
            "Train_Time_Sec": metrics["train_time"],
            "Predict_Time_Sec": metrics["predict_time"],
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)