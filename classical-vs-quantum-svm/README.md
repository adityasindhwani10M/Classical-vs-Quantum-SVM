# Classical SVM vs Quantum SVM: A Comparative Study Using Quantum Kernels

This research-based project compares the classification performance, training time, and prediction time of **Classical Support Vector Machines (SVM)** against **Quantum Support Vector Machines (QSVM)** using quantum kernels.

The project evaluates both paradigms across three benchmark datasets:
1. **Iris Dataset** (3 classes, 4 features)
2. **Wine Dataset** (3 classes, 13 features)
3. **Breast Cancer Wisconsin Dataset** (2 classes, 30 features)

---

## 🧬 Theoretical Overview

### Classical SVM
Classical Support Vector Machines use mathematical functions (kernels) like RBF (Radial Basis Function) or Linear kernels to project data into a higher-dimensional space where classes are linearly separable.

### Quantum SVM (QSVC)
Quantum Support Vector Classification utilizes a quantum computer (or classical simulator) to evaluate a **quantum kernel**. 
1. **Feature Map**: Classical data $x$ is mapped to a quantum state $|\phi(x)\rangle$ using a parameterized quantum circuit (here, `zz_feature_map`).
2. **Quantum Kernel**: The kernel value $K(x_i, x_j)$ represents the transition probability (fidelity) between the two states:
   $$K(x_i, x_j) = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$$
3. The fidelity is computed using Qiskit's `ComputeUncompute` algorithm via local simulation (`StatevectorSampler`).

---

## 📁 Project Structure

```
classical-vs-quantum-svm/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Loads Iris, Wine, and Breast Cancer datasets from sklearn
│   ├── preprocessing.py     # Downsampling, Standard Scaling, PCA, and MinMaxScaler to [-1, 1]
│   ├── classical_svm.py     # Training/predicting classical SVM (RBF & Linear)
│   ├── quantum_svm.py       # Training/predicting QSVC using Qiskit 2.x StatevectorSampler
│   ├── evaluation.py        # Computes Accuracy, F1, Precision, Recall, and Confusion Matrices
│   └── visualization.py     # Generates bar charts and confusion matrix heatmaps
│
├── datasets/                # Placeholders for custom dataset exports
├── results/                 # Stores generated evaluation metrics in CSV format
├── graphs/                  # Stores generated comparison charts and heatmaps
├── report/                  # Stores the final comparison Markdown research report
├── main.py                  # Experiment orchestrator and entry point
├── requirements.txt         # Project dependencies
└── README.md                # Setup and usage instructions
```

---

## 🚀 Getting Started

### Prerequisites
* **Python**: Version `3.11.x` is recommended.
* **Pip**: Python package installer.

### Setup and Installation

1. Navigate to the project root directory:
   ```powershell
   cd C:\Users\ASUS\.gemini\antigravity-ide\scratch\classical-vs-quantum-svm
   ```

2. Create a Python virtual environment:
   ```powershell
   python -m venv .venv
   ```

3. Activate the virtual environment:
   * **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Windows (cmd)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **Linux/macOS**:
     ```bash
     source .venv/bin/activate
     ```

4. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🏃 Execution

To run the full benchmark experiment suite and generate the comparative plots and research report:

```powershell
python main.py
```

### Note on Execution Time & Settings
* **Default Downsampling**: The script downsamples datasets to $100$ samples to ensure the quantum simulations execute within ~1.5 minutes.
* **PCA Dimension Reduction**: Features are reduced to $2$ dimensions (mapped to $2$ qubits) using PCA. 
* To run the study with more qubits or larger sample sizes, you can adjust `max_samples` and `n_components` variables in [main.py](file:///C:/Users/ASUS/.gemini/antigravity-ide/scratch/classical-vs-quantum-svm/main.py).

---

## 📊 Outputs & Results

After execution:
* **CSV Metrics Table**: Saved as `results/comparison_results.csv`.
* **Visual Plots**: Bar charts comparing model accuracy, F1 score, and training times, as well as confusion matrix heatmaps are saved in the `graphs/` folder.
* **Final Report**: The final Markdown summary report with research observations and NISQ hardware challenges is written to `report/comparison_report.md`.
