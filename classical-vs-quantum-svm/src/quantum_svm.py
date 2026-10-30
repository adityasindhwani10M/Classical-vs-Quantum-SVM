import time
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler
# pyrefly: ignore [missing-import]
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

def train_quantum_svm(X_train, y_train, reps=2, entanglement='linear'):
    """
    Trains a Quantum Support Vector Classifier (QSVC) using a Fidelity Quantum Kernel.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training labels.
    reps : int
        Number of repetitions of the feature map (default 2).
    entanglement : str
        Type of entanglement in the ZZFeatureMap ('linear', 'circular', 'full').
        
    Returns:
    --------
    model : qiskit_machine_learning.algorithms.QSVC
        The trained QSVC model.
    train_time : float
        Time taken to compute the kernel matrix and fit the model in seconds.
    """
    n_features = X_train.shape[1]
    
    # 1. Define the quantum feature map using the functional form for Qiskit 2.x
    if n_features <= 1:
        from qiskit.circuit.library import z_feature_map
        feature_map = z_feature_map(feature_dimension=n_features, reps=reps)
    else:
        feature_map = zz_feature_map(feature_dimension=n_features, reps=reps, entanglement=entanglement)
    
    # 2. Set up the StatevectorSampler primitive and the ComputeUncompute fidelity
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    
    # 3. Create the Quantum Kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    
    # 4. Instantiate QSVC (extends scikit-learn's SVC)
    model = QSVC(quantum_kernel=quantum_kernel)
    
    # 5. Train model and measure time
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    end_time = time.perf_counter()
    
    train_time = end_time - start_time
    return model, train_time

def predict_quantum_svm(model, X_test):
    """
    Predicts using the trained QSVC model and measures prediction time.
    
    Parameters:
    -----------
    model : qiskit_machine_learning.algorithms.QSVC
        Trained QSVC model.
    X_test : numpy.ndarray
        Testing features.
        
    Returns:
    --------
    y_pred : numpy.ndarray
        Predicted labels.
    predict_time : float
        Time taken for prediction in seconds.
    """
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time = time.perf_counter()
    
    predict_time = end_time - start_time
    return y_pred, predict_time
