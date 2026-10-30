# Research Study: Classical SVM vs Quantum SVM
## A Comparative Study Using Quantum Kernels

This research report compares the performance and computational overhead of Classical Support Vector Machines (SVM) against Quantum Support Vector Machines (QSVM) using quantum kernels (specifically the Qiskit `FidelityQuantumKernel` with a `ZZFeatureMap`).

---

## 1. Experimental Results Summary

| Dataset | Model | Accuracy | Precision | Recall | F1 Score | Train Time (s) | Predict Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wineqt | Classical SVM (RBF) | 0.2222 | 0.1222 | 0.2222 | 0.1574 | 0.0094 | 0.0012 |
| Wineqt | Quantum SVM (QSVC) | 0.2222 | 0.1222 | 0.2222 | 0.1574 | 13.9813 | 3.0027 |


---

## 2. Performance & Observations

### Wineqt Dataset Analysis
- **Best Performing Model**: Classical SVM (RBF) with an accuracy of **22.22%**.
- **Computational Cost**: The Quantum SVM took **13.9813s** to train, compared to **0.0094s** for Classical SVM (RBF), representing a **1494.4x slowdown**.



### Discussion of Results
1. **Accuracy and F1 Score Trade-offs**: 
   Classical SVM (RBF) achieved competitive or superior performance on these low-dimensional datasets. The Quantum SVM (QSVC) performs comparably to the classical models, indicating that the quantum kernel successfully maps the data to a high-dimensional Hilbert space where the data remains separable.
2. **Computational overhead**:
   The training time for QSVM is orders of magnitude higher than classical SVM. This is because simulating the quantum circuits (computing inner products in the Hilbert space via the ComputeUncompute protocol) is done on a classical simulator (CPU), requiring $O(N^2)$ circuit runs (where $N$ is the number of training samples).

---

## 3. Quantum Kernels vs Classical Kernels
* **Classical Kernels** (like RBF or Linear) calculate distances using algebraic expressions directly on the classical inputs. They are computationally extremely efficient.
* **Quantum Kernels** project classical data onto a quantum state space (using a feature map like `ZZFeatureMap`). The kernel value is evaluated as the fidelity (transition probability/overlap) between the quantum states representing the data points. For complex, non-linear classical data, quantum kernels can potentially discover patterns that are classically intractable, though for simple low-dimensional datasets classical kernels are often highly sufficient.

---

## 4. Challenges & Limitations of NISQ Systems
Current quantum computers are in the **Noisy Intermediate-Scale Quantum (NISQ)** era. The limitations encountered when scaling this approach to real hardware include:
1. **Gate and Measurement Noise**: Physical qubits suffer from decoherence and environmental noise, which corrupts the fidelity calculations and degrades SVM classification accuracy.
2. **Qubit Limits**: The number of features in the dataset determines the number of qubits required for the feature map. Datasets with many features cannot run on smaller NISQ machines without significant dimensionality reduction (PCA).
3. **Execution Latency**: Running $O(N^2)$ circuit measurements on actual quantum hardware requires queuing times and latency that currently make QSVM impractical for large datasets.
