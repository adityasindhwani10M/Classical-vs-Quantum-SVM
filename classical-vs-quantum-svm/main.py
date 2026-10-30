import argparse
import os
import pandas as pd

from src.dataset_loader import load_uploaded_dataset
from src.preprocessing import preprocess_dataset
from src.classical_svm import train_classical_svm, predict_classical_svm
from src.quantum_svm import train_quantum_svm, predict_quantum_svm
from src.evaluation import evaluate_predictions, save_results_to_csv
from src.visualization import plot_performance_metrics, plot_confusion_matrices


def run_comparative_study(dataset_paths, target_column=None, max_samples=100, n_components=2, max_load_rows=10000, skip_quantum=False):
    print("=" * 70)
    print("  CLASSICAL SVM VS QUANTUM SVM: A COMPARATIVE STUDY USING QUANTUM KERNELS  ")
    print("=" * 70)

    results = {}

    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        print(f"\n>>> Running Experiments on [{dataset_name}] Dataset...")

        with open(dataset_path, "rb") as file_obj:
            X, y, feature_names, target_names, dataset_key = load_uploaded_dataset(
                file_obj,
                target_column=target_column,
                dataset_name=dataset_name,
                max_rows=max_load_rows,
            )

        print(f"Loaded {len(y)} samples with {X.shape[1]} features and {len(target_names)} classes.")

        X_train, X_test, y_train, y_test = preprocess_dataset(
            X, y, test_size=0.2, max_samples=max_samples, n_components=n_components, random_state=42
        )
        print(f"Preprocessed: Train set = {X_train.shape[0]} samples, Test set = {X_test.shape[0]} samples.")
        print(f"Features reduced via PCA from {X.shape[1]} to {X_train.shape[1]} dimensions.")

        print("- Training Classical SVM (RBF Kernel)...")
        try:
            c_rbf_model, c_rbf_train_time = train_classical_svm(X_train, y_train, kernel="rbf")
            c_rbf_preds, c_rbf_pred_time = predict_classical_svm(c_rbf_model, X_test)
            c_rbf_metrics = evaluate_predictions(y_test, c_rbf_preds, c_rbf_train_time, c_rbf_pred_time)
            results[(dataset_key, "Classical SVM (RBF)")] = c_rbf_metrics
            print(f"  RBF Accuracy: {c_rbf_metrics['accuracy']:.4f} | Train Time: {c_rbf_train_time:.5f}s")
        except Exception as e:
            print(f"  [ERROR] Classical SVM (RBF) failed on {dataset_key}: {e}")

        print("- Training Classical SVM (Linear Kernel)...")
        try:
            c_lin_model, c_lin_train_time = train_classical_svm(X_train, y_train, kernel="linear")
            c_lin_preds, c_lin_pred_time = predict_classical_svm(c_lin_model, X_test)
            c_lin_metrics = evaluate_predictions(y_test, c_lin_preds, c_lin_train_time, c_lin_pred_time)
            results[(dataset_key, "Classical SVM (Linear)")] = c_lin_metrics
            print(f"  Linear Accuracy: {c_lin_metrics['accuracy']:.4f} | Train Time: {c_lin_train_time:.5f}s")
        except Exception as e:
            print(f"  [ERROR] Classical SVM (Linear) failed on {dataset_key}: {e}")

        if not skip_quantum:
            print("- Training Quantum SVM (QSVC with ZZFeatureMap)...")
            try:
                q_model, q_train_time = train_quantum_svm(X_train, y_train, reps=2, entanglement="linear")
                q_preds, q_pred_time = predict_quantum_svm(q_model, X_test)
                q_metrics = evaluate_predictions(y_test, q_preds, q_train_time, q_pred_time)
                results[(dataset_key, "Quantum SVM (QSVC)")] = q_metrics
                print(f"  QSVC Accuracy: {q_metrics['accuracy']:.4f} | Train Time: {q_train_time:.5f}s")
            except Exception as e:
                print(f"  [ERROR] Quantum SVM failed on {dataset_key}: {e}")
        else:
            print("- Skipping Quantum SVM training as requested.")

    print("\n" + "=" * 70)
    print("  EXPERIMENTS COMPLETED  ")
    print("=" * 70)

    if not results:
        print("\n[ERROR] No models were trained successfully for any dataset.")
        return

    csv_path = os.path.join("results", "comparison_results.csv")
    save_results_to_csv(results, csv_path)

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

    print("\nSummary Table:")
    print(df.to_string(index=False, formatters={
        "Accuracy": "{:,.4f}".format,
        "Precision": "{:,.4f}".format,
        "Recall": "{:,.4f}".format,
        "F1_Score": "{:,.4f}".format,
        "Train_Time_Sec": "{:,.4f}s".format,
        "Predict_Time_Sec": "{:,.4f}s".format,
    }))

    print("\nGenerating visual graphs...")
    plot_performance_metrics(df, "graphs")
    plot_confusion_matrices(results, "graphs")
    print("Graphs saved under graphs/ directory.")

    generate_markdown_report(df, results)
    print("Research report generated: report/comparison_report.md")


def generate_markdown_report(df, results_dict):
    os.makedirs("report", exist_ok=True)
    report_path = os.path.join("report", "comparison_report.md")

    md_table = "| Dataset | Model | Accuracy | Precision | Recall | F1 Score | Train Time (s) | Predict Time (s) |\n"
    md_table += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for _, r in df.iterrows():
        md_table += (
            f"| {r['Dataset'].capitalize()} | {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | "
            f"{r['Recall']:.4f} | {r['F1_Score']:.4f} | {r['Train_Time_Sec']:.4f} | {r['Predict_Time_Sec']:.4f} |\n"
        )

    observations_section = ""
    for dataset in df["Dataset"].unique():
        models = df[df["Dataset"] == dataset]
        observations_section += f"### {dataset.capitalize()} Dataset Analysis\n"
        if not models.empty:
            best_model = models.loc[models["Accuracy"].idxmax()]["Model"]
            best_acc = models["Accuracy"].max()
            observations_section += f"- **Best Performing Model**: {best_model} with an accuracy of **{best_acc:.2%}**.\n"

            q_row = models[models["Model"].str.contains("Quantum")]
            c_rbf_row = models[models["Model"].str.contains("RBF")]
            if not q_row.empty and not c_rbf_row.empty:
                q_time = q_row.iloc[0]["Train_Time_Sec"]
                c_time = c_rbf_row.iloc[0]["Train_Time_Sec"]
                slowdown = q_time / c_time if c_time > 0 else float("inf")
                observations_section += (
                    f"- **Computational Cost**: The Quantum SVM took **{q_time:.4f}s** to train, compared to "
                    f"**{c_time:.4f}s** for Classical SVM (RBF), representing a **{slowdown:.1f}x slowdown**.\n"
                )
        observations_section += "\n"

    report_content = f"""# Research Study: Classical SVM vs Quantum SVM
## A Comparative Study Using Quantum Kernels

This research report compares the performance and computational overhead of Classical Support Vector Machines (SVM) against Quantum Support Vector Machines (QSVM) using quantum kernels (specifically the Qiskit `FidelityQuantumKernel` with a `ZZFeatureMap`).

---

## 1. Experimental Results Summary

{md_table}

---

## 2. Performance & Observations

{observations_section}

### Discussion of Results
1. **Accuracy and F1 Score Trade-offs**:
   Classical SVM with RBF and Linear kernels achieves competitive or superior performance on many low-dimensional datasets. The Quantum SVM (QSVC) performs comparably to the classical models, indicating that the quantum kernel successfully maps the data to a high-dimensional Hilbert space where the data remains separable.
2. **Computational overhead**:
   The training time for QSVM is orders of magnitude higher than classical SVM. This is because simulating the quantum circuits (computing inner products in the Hilbert space via the ComputeUncompute protocol) is done on a classical simulator (CPU), requiring $O(N^2)$ circuit runs (where $N$ is the number of training samples).

---

## 3. Quantum Kernels vs Classical Kernels
* **Classical Kernels** (like RBF or Linear) calculate distances using algebraic expressions directly on the classical inputs. They are computationally extremely efficient.
* **Quantum Kernels** project classical data onto a quantum state space (using a feature map like `ZZFeatureMap`). The kernel value is evaluated as the fidelity (transition probability/overlap) between the quantum states representing the data points.

---

## 4. Challenges & Limitations of NISQ Systems
Current quantum computers are in the **Noisy Intermediate-Scale Quantum (NISQ)** era. The limitations encountered when scaling this approach to real hardware include:
1. **Gate and Measurement Noise**: Physical qubits suffer from decoherence and environmental noise, which corrupts the fidelity calculations and degrades SVM classification accuracy.
2. **Qubit Limits**: The number of features in the dataset determines the number of qubits required for the feature map. Datasets with many features cannot run on smaller NISQ machines without significant dimensionality reduction (PCA).
3. **Execution Latency**: Running $O(N^2)$ circuit measurements on actual quantum hardware requires queuing times and latency that currently make QSVM impractical for large datasets.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Classical vs Quantum SVM comparison on uploaded CSV datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Paths to one or more CSV dataset files.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Name of the target/label column. Defaults to the last column in each CSV.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to use per dataset.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=2,
        help="Number of PCA components (qubits) to use.",
    )
    parser.add_argument(
        "--max-load-rows",
        type=int,
        default=10000,
        help="Maximum rows to load from each CSV file to prevent memory issues.",
    )
    parser.add_argument(
        "--skip-quantum",
        action="store_true",
        help="Skip training the Quantum SVM model.",
    )
    args = parser.parse_args()

    run_comparative_study(
        dataset_paths=args.datasets,
        target_column=args.target_column,
        max_samples=args.max_samples,
        n_components=args.pca_components,
        max_load_rows=args.max_load_rows,
        skip_quantum=args.skip_quantum,
    )
