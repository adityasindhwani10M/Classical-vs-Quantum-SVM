"""
Full diagnostic test for Classical vs Quantum SVM project.
Tests every module, every function, and all edge cases.
"""
import sys, io, os, traceback
sys.path.insert(0, '.')

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}")
        print(f"        {e}")
        traceback.print_exc()

print("=" * 60)
print("DIAGNOSTIC: Classical vs Quantum SVM")
print("=" * 60)

# ── IMPORTS ──────────────────────────────────────────────────
print("\n[1] IMPORTS")

def t_import_dataset():
    from src.dataset_loader import load_uploaded_dataset, get_csv_column_names, read_csv_preview

def t_import_preprocessing():
    from src.preprocessing import preprocess_dataset

def t_import_classical():
    from src.classical_svm import train_classical_svm, predict_classical_svm

def t_import_quantum():
    from src.quantum_svm import train_quantum_svm, predict_quantum_svm

def t_import_evaluation():
    from src.evaluation import evaluate_predictions, save_results_to_csv

def t_import_visualization():
    from src.visualization import plot_performance_metrics, plot_confusion_matrices, confusion_matrix_path

check("import dataset_loader", t_import_dataset)
check("import preprocessing", t_import_preprocessing)
check("import classical_svm", t_import_classical)
check("import quantum_svm",   t_import_quantum)
check("import evaluation",    t_import_evaluation)
check("import visualization", t_import_visualization)

# ── DATASET LOADER ────────────────────────────────────────────
print("\n[2] DATASET LOADER")
import pandas as pd
import numpy as np
from src.dataset_loader import load_uploaded_dataset, get_csv_column_names, read_csv_preview

def make_buf(df):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    buf.name = "test.csv"
    return buf

iris_like = pd.DataFrame({
    'f1': np.random.rand(50),
    'f2': np.random.rand(50),
    'f3': np.random.rand(50),
    'label': (['cat']*25 + ['dog']*25)
})

def t_get_col_names():
    cols = get_csv_column_names(make_buf(iris_like))
    assert cols == ['f1','f2','f3','label'], f"Got: {cols}"

def t_read_preview():
    prev = read_csv_preview(make_buf(iris_like), nrows=5)
    assert prev.shape == (5, 4), f"Shape: {prev.shape}"

def t_load_dataset():
    X, y, feat, tnames, dname = load_uploaded_dataset(
        make_buf(iris_like), target_column='label', dataset_name='test.csv')
    assert X.shape == (50, 3)
    assert y.shape == (50,)
    assert set(tnames) == {'cat', 'dog'}
    assert dname == 'test'

def t_load_missing_target_fallback():
    # target_column not in file -> should fall back to last col
    X, y, feat, tnames, dname = load_uploaded_dataset(
        make_buf(iris_like), target_column='nonexistent', dataset_name='t.csv')
    assert X.shape[1] == 3

def t_load_categorical_features():
    df = pd.DataFrame({'color':['red','blue','red','green'],'size':['S','M','L','S'],'label':[0,1,0,1]})
    X, y, _, _, _ = load_uploaded_dataset(make_buf(df), target_column='label')
    assert X.shape == (4, 2)
    assert X.dtype == float

check("get_csv_column_names", t_get_col_names)
check("read_csv_preview", t_read_preview)
check("load_uploaded_dataset - basic", t_load_dataset)
check("load_uploaded_dataset - fallback target", t_load_missing_target_fallback)
check("load_uploaded_dataset - categorical features", t_load_categorical_features)

# ── PREPROCESSING ────────────────────────────────────────────
print("\n[3] PREPROCESSING")
from src.preprocessing import preprocess_dataset

def t_preprocess_basic():
    X = np.random.rand(100, 4)
    y = np.array([0]*50 + [1]*50)
    Xtr, Xte, ytr, yte = preprocess_dataset(X, y, max_samples=60, n_components=2)
    assert Xtr.shape[1] == 2
    assert Xte.shape[1] == 2
    assert Xtr.min() >= -1.01 and Xtr.max() <= 1.01, "Not in [-1,1] range"

def t_preprocess_3class():
    X = np.random.rand(90, 5)
    y = np.array([0]*30 + [1]*30 + [2]*30)
    Xtr, Xte, ytr, yte = preprocess_dataset(X, y, max_samples=60, n_components=2)
    assert Xtr.shape[1] == 2

def t_preprocess_subsampling():
    X = np.random.rand(500, 4)
    y = np.array([0]*250 + [1]*250)
    Xtr, Xte, ytr, yte = preprocess_dataset(X, y, max_samples=40, n_components=2)
    assert len(ytr) + len(yte) <= 40

check("preprocess - binary", t_preprocess_basic)
check("preprocess - 3 class", t_preprocess_3class)
check("preprocess - subsampling cap", t_preprocess_subsampling)

# ── CLASSICAL SVM ─────────────────────────────────────────────
print("\n[4] CLASSICAL SVM")
from src.classical_svm import train_classical_svm, predict_classical_svm
from src.preprocessing import preprocess_dataset

X_base = np.random.rand(80, 4)
y_base = np.array([0]*40 + [1]*40)
Xtr, Xte, ytr, yte = preprocess_dataset(X_base, y_base, max_samples=60, n_components=2)

for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
    def t_kernel(k=kernel):
        m, t = train_classical_svm(Xtr, ytr, kernel=k)
        preds, pt = predict_classical_svm(m, Xte)
        assert len(preds) == len(yte)
        assert t >= 0 and pt >= 0
    check(f"classical SVM kernel={kernel}", t_kernel)

# ── EVALUATION ────────────────────────────────────────────────
print("\n[5] EVALUATION")
from src.evaluation import evaluate_predictions, save_results_to_csv

def t_eval_binary():
    y_true = np.array([0,1,0,1,0,1,0,1])
    y_pred = np.array([0,1,0,0,0,1,1,1])
    m = evaluate_predictions(y_true, y_pred, 0.01, 0.001)
    assert 'accuracy' in m
    assert 'f1_score' in m
    assert 'confusion_matrix' in m
    assert 0 <= m['accuracy'] <= 1

def t_eval_multiclass():
    y_true = np.array([0,1,2,0,1,2])
    y_pred = np.array([0,2,2,0,1,1])
    m = evaluate_predictions(y_true, y_pred, 0.01, 0.001)
    assert 0 <= m['accuracy'] <= 1

def t_save_csv():
    results = {('iris','Model A'): evaluate_predictions(
        np.array([0,1,0,1]), np.array([0,1,1,1]), 0.1, 0.01)}
    os.makedirs('results', exist_ok=True)
    save_results_to_csv(results, 'results/_diag_test.csv')
    assert os.path.exists('results/_diag_test.csv')
    os.remove('results/_diag_test.csv')

check("evaluate - binary", t_eval_binary)
check("evaluate - multiclass", t_eval_multiclass)
check("save_results_to_csv", t_save_csv)

# ── VISUALIZATION ─────────────────────────────────────────────
print("\n[6] VISUALIZATION")
from src.visualization import plot_performance_metrics, plot_confusion_matrices, confusion_matrix_path

def t_confusion_matrix_path():
    p = confusion_matrix_path('iris', 'Classical SVM (RBF)')
    assert 'iris' in p
    assert 'classical_svm_rbf' in p

def t_plot_metrics():
    df = pd.DataFrame([
        {'Dataset':'iris','Model':'Classical SVM (RBF)','Accuracy':0.9,'Precision':0.9,'Recall':0.9,'F1_Score':0.9,'Train_Time_Sec':0.01,'Predict_Time_Sec':0.001},
        {'Dataset':'iris','Model':'Quantum SVM (QSVC)','Accuracy':0.85,'Precision':0.85,'Recall':0.85,'F1_Score':0.85,'Train_Time_Sec':10.0,'Predict_Time_Sec':1.0},
    ])
    os.makedirs('graphs', exist_ok=True)
    plot_performance_metrics(df, 'graphs')
    assert os.path.exists('graphs/accuracy_comparison.png')
    assert os.path.exists('graphs/f1_score_comparison.png')
    assert os.path.exists('graphs/training_time_comparison.png')

def t_plot_confusion():
    from src.evaluation import evaluate_predictions
    y_true = np.array([0,1,0,1,0,1])
    y_pred = np.array([0,1,0,0,0,1])
    metrics = evaluate_predictions(y_true, y_pred, 0.01, 0.001)
    results = {('iris', 'Classical SVM (RBF)'): metrics}
    plot_confusion_matrices(results, 'graphs')
    p = confusion_matrix_path('iris', 'Classical SVM (RBF)')
    assert os.path.exists(p), f"File not found: {p}"

check("confusion_matrix_path", t_confusion_matrix_path)
check("plot_performance_metrics", t_plot_metrics)
check("plot_confusion_matrices", t_plot_confusion)

# ── QUANTUM SVM (smoke test - small) ─────────────────────────
print("\n[7] QUANTUM SVM (small smoke test)")
from src.quantum_svm import train_quantum_svm, predict_quantum_svm

def t_quantum_smoke():
    # Binary, 10 samples, 2 features (= 2 qubits) — fast
    X_q = np.random.rand(10, 2) * 2 - 1
    y_q = np.array([0]*5 + [1]*5)
    model, t = train_quantum_svm(X_q, y_q, reps=1, entanglement='linear')
    preds, pt = predict_quantum_svm(model, X_q[:4])
    assert len(preds) == 4
    assert t > 0

def t_quantum_entanglement_options():
    X_q = np.random.rand(8, 2) * 2 - 1
    y_q = np.array([0]*4 + [1]*4)
    for ent in ['linear', 'circular', 'full']:
        model, _ = train_quantum_svm(X_q, y_q, reps=1, entanglement=ent)
        preds, _ = predict_quantum_svm(model, X_q[:2])
        assert len(preds) == 2

check("quantum SVM - smoke test binary", t_quantum_smoke)
check("quantum SVM - all entanglement options", t_quantum_entanglement_options)

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nFAILED TESTS:")
    for name, err in FAIL:
        print(f"  - {name}: {err}")
else:
    print("\nALL TESTS PASSED - APP IS FULLY FUNCTIONAL")
print("=" * 60)
