import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from src.dataset_loader import load_uploaded_dataset
from src.preprocessing import preprocess_dataset
from src.classical_svm import train_classical_svm, predict_classical_svm
from src.quantum_svm import train_quantum_svm, predict_quantum_svm
from src.evaluation import evaluate_predictions, save_results_to_csv
from src.visualization import plot_performance_metrics, plot_confusion_matrices

print("=== Testing all modules end-to-end ===\n")

# 1. Dataset loading
print("1. Loading dataset...")
X, y, feature_names, target_names, dataset_name = load_uploaded_dataset(
    "test_data/iris.csv",
    target_column="target",
    dataset_name="iris"
)
print(f"   OK: X={X.shape}, y={y.shape}, classes={target_names}, name={dataset_name}")

# 2. Preprocessing
print("\n2. Preprocessing...")
X_train, X_test, y_train, y_test = preprocess_dataset(
    X, y, test_size=0.2, max_samples=60, n_components=2, random_state=42
)
print(f"   OK: Train={X_train.shape}, Test={X_test.shape}")

# 3. Classical SVM RBF
print("\n3. Training Classical SVM (RBF)...")
model_rbf, t_rbf = train_classical_svm(X_train, y_train, kernel="rbf")
preds_rbf, tp_rbf = predict_classical_svm(model_rbf, X_test)
metrics_rbf = evaluate_predictions(y_test, preds_rbf, t_rbf, tp_rbf)
print(f"   OK: Accuracy={metrics_rbf['accuracy']:.4f}, F1={metrics_rbf['f1_score']:.4f}, Train={t_rbf:.4f}s")

# 4. Classical SVM Linear
print("\n4. Training Classical SVM (LINEAR)...")
model_lin, t_lin = train_classical_svm(X_train, y_train, kernel="linear")
preds_lin, tp_lin = predict_classical_svm(model_lin, X_test)
metrics_lin = evaluate_predictions(y_test, preds_lin, t_lin, tp_lin)
print(f"   OK: Accuracy={metrics_lin['accuracy']:.4f}, F1={metrics_lin['f1_score']:.4f}, Train={t_lin:.4f}s")

# 5. Quantum SVM
print("\n5. Training Quantum SVM (QSVC) — may take 1-3 min...")
try:
    model_q, t_q = train_quantum_svm(X_train, y_train, reps=1, entanglement="linear")
    preds_q, tp_q = predict_quantum_svm(model_q, X_test)
    metrics_q = evaluate_predictions(y_test, preds_q, t_q, tp_q)
    print(f"   OK: Accuracy={metrics_q['accuracy']:.4f}, F1={metrics_q['f1_score']:.4f}, Train={t_q:.2f}s")
    quantum_ok = True
except Exception as e:
    print(f"   FAIL: {e}")
    metrics_q = None
    quantum_ok = False

# 6. Save CSV
print("\n6. Saving results CSV...")
results = {
    ("iris", "Classical SVM (RBF)"): metrics_rbf,
    ("iris", "Classical SVM (LINEAR)"): metrics_lin,
}
if quantum_ok:
    results[("iris", "Quantum SVM (QSVC)")] = metrics_q
os.makedirs("results", exist_ok=True)
save_results_to_csv(results, "results/comparison_results.csv")
print("   OK: results/comparison_results.csv written")

# 7. Build DataFrame
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
print(f"\n   Results DataFrame:\n{df.to_string()}")

# 8. Performance charts
print("\n7. Generating performance charts...")
os.makedirs("graphs", exist_ok=True)
plot_performance_metrics(df, "graphs")
charts = ["accuracy_comparison.png", "f1_score_comparison.png", "training_time_comparison.png"]
for c in charts:
    p = os.path.join("graphs", c)
    status = "OK" if os.path.exists(p) else "MISSING"
    print(f"   {status}: {c}")

# 9. Confusion matrices
print("\n8. Generating confusion matrices...")
plot_confusion_matrices(results, "graphs")
cm_files = [f for f in os.listdir("graphs") if f.startswith("confusion_matrix")]
print(f"   OK: {len(cm_files)} confusion matrix files created: {cm_files}")

# 10. Report generation test
print("\n9. Report generation check...")
import re
report_path = os.path.join("report", "comparison_report.md")
os.makedirs("report", exist_ok=True)
md_table = "| Dataset | Model | Accuracy |\n| --- | --- | --- |\n"
for _, r in df.iterrows():
    md_table += f"| {r['Dataset']} | {r['Model']} | {r['Accuracy']:.4f} |\n"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Test Report\n\n" + md_table)
print(f"   OK: report written to {report_path}")

print("\n=== SUMMARY ===")
print(f"  Classical SVM (RBF):    Accuracy={metrics_rbf['accuracy']:.4f}")
print(f"  Classical SVM (LINEAR): Accuracy={metrics_lin['accuracy']:.4f}")
if quantum_ok:
    print(f"  Quantum SVM (QSVC):     Accuracy={metrics_q['accuracy']:.4f}")
else:
    print("  Quantum SVM (QSVC):     FAILED")
print("\n=== All tests complete ===")
