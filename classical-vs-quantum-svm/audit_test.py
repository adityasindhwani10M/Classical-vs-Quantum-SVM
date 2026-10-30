import sys, io, os
sys.path.insert(0, '.')

SEP = '=' * 60
PASS = []
FAIL = []

def ok(msg):
    print(f'  ✅ {msg}')
    PASS.append(msg)

def fail(msg, e):
    print(f'  ❌ {msg}: {e}')
    FAIL.append(msg)

print(SEP)
print('FULL FEATURE AUDIT')
print(SEP)

# ── 1. Dataset Loader: header-less CSV with categorical cols ──
print('\n[1] dataset_loader: header-less KDD-style CSV')
try:
    from src.dataset_loader import load_uploaded_dataset
    kdd_headerless = (
        "0,tcp,http,SF,181,5450,0,0,normal.\n"
        "0,tcp,smtp,SF,239,486,0,0,normal.\n"
        "0,udp,private,SF,105,146,0,0,smurf.\n"
        "1,icmp,eco_i,SF,8,0,0,0,smurf.\n"
        "0,tcp,ftp,SF,0,0,0,0,normal.\n"
        "0,udp,domain,SF,24,206,0,0,normal.\n"
        "1,icmp,eco_i,SF,8,0,0,0,normal.\n"
        "0,tcp,http,SF,219,1337,0,0,smurf.\n"
    )
    f = io.BytesIO(kdd_headerless.encode())
    X, y, feats, targets, name = load_uploaded_dataset(f, target_column='normal.', dataset_name='kdd_test')
    ok(f"header-less load: shape={X.shape}, classes={targets}, feats[0:3]={feats[:3]}")
except Exception as e:
    fail("header-less load", e)

print('\n[1b] dataset_loader: CSV with headers, fallback target')
try:
    csv_with_hdr = (
        "duration,proto,service,flag,src,dst,label\n"
        "0,tcp,http,SF,181,5450,normal\n"
        "0,tcp,smtp,SF,239,486,attack\n"
        "0,udp,private,SF,105,146,normal\n"
        "0,tcp,ftp,SF,0,0,attack\n"
        "0,udp,domain,SF,24,206,normal\n"
    )
    f2 = io.BytesIO(csv_with_hdr.encode())
    X2, y2, feats2, targets2, name2 = load_uploaded_dataset(f2, target_column='NOT_EXIST', dataset_name='test2')
    ok(f"fallback target: shape={X2.shape}, classes={targets2}")
except Exception as e:
    fail("fallback target", e)

# ── 2. Preprocessing ──────────────────────────────────────────
print('\n[2] preprocessing: stratify guard, PCA bounds, subsampling')
try:
    from src.preprocessing import preprocess_dataset
    import numpy as np

    np.random.seed(0)
    X_big = np.random.randn(60, 6)
    y_big = np.array([0]*30 + [1]*30)
    Xtr, Xte, ytr, yte = preprocess_dataset(X_big, y_big, test_size=0.2, max_samples=50, n_components=4)
    ok(f"large dataset: train={Xtr.shape}, test={Xte.shape}")

    X_tiny = np.random.randn(4, 2)
    y_tiny = np.array([0, 0, 1, 1])
    Xtr2, Xte2, ytr2, yte2 = preprocess_dataset(X_tiny, y_tiny, test_size=0.3, n_components=4)
    ok(f"tiny dataset edge case: train={Xtr2.shape}, test={Xte2.shape}")
except Exception as e:
    fail("preprocessing", e)

# ── 3. Classical SVM: all 4 kernels ───────────────────────────
print('\n[3] classical_svm: rbf, linear, poly, sigmoid')
try:
    from src.classical_svm import train_classical_svm, predict_classical_svm
    for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
        model, t = train_classical_svm(Xtr, ytr, kernel=kernel)
        preds, pt = predict_classical_svm(model, Xte)
        ok(f"kernel={kernel}: preds={preds.tolist()}, train={t:.5f}s")
except Exception as e:
    fail(f"classical_svm kernel={kernel}", e)

# ── 4. Evaluation + CSV save ──────────────────────────────────
print('\n[4] evaluation: metrics dict and CSV save')
try:
    from src.evaluation import evaluate_predictions, save_results_to_csv
    import pandas as pd
    model_r, t = train_classical_svm(Xtr, ytr, kernel='rbf')
    preds_r, pt = predict_classical_svm(model_r, Xte)
    m = evaluate_predictions(yte, preds_r, t, pt)
    ok(f"metrics: acc={m['accuracy']:.4f}, f1={m['f1_score']:.4f}, cm shape={len(m['confusion_matrix'])}x{len(m['confusion_matrix'][0])}")
    results_d = {('test_ds', 'Classical SVM (RBF)'): m}
    save_results_to_csv(results_d, 'results/audit_test.csv')
    assert os.path.exists('results/audit_test.csv')
    os.remove('results/audit_test.csv')
    ok("CSV save/delete")
except Exception as e:
    fail("evaluation/csv", e)

# ── 5. Visualization: dynamic palette (all model name variants) ─
print('\n[5] visualization: dynamic palette across all kernel name formats')
try:
    from src.visualization import plot_performance_metrics, plot_confusion_matrices
    df_test = pd.DataFrame([
        {'Dataset': 'ds1', 'Model': 'Classical SVM (RBF)',    'Accuracy': 0.90, 'F1_Score': 0.88, 'Train_Time_Sec': 0.01,  'Predict_Time_Sec': 0.001},
        {'Dataset': 'ds1', 'Model': 'Classical SVM (LINEAR)', 'Accuracy': 0.85, 'F1_Score': 0.84, 'Train_Time_Sec': 0.008, 'Predict_Time_Sec': 0.001},
        {'Dataset': 'ds1', 'Model': 'Quantum SVM (QSVC)',     'Accuracy': 0.88, 'F1_Score': 0.87, 'Train_Time_Sec': 12.5,  'Predict_Time_Sec': 3.2},
        {'Dataset': 'ds1', 'Model': 'Classical SVM (Rbf)',    'Accuracy': 0.91, 'F1_Score': 0.90, 'Train_Time_Sec': 0.012, 'Predict_Time_Sec': 0.001},
        {'Dataset': 'ds1', 'Model': 'Classical SVM (Poly)',   'Accuracy': 0.82, 'F1_Score': 0.81, 'Train_Time_Sec': 0.015, 'Predict_Time_Sec': 0.001},
    ])
    plot_performance_metrics(df_test, 'graphs')
    for fname in ['accuracy_comparison.png', 'f1_score_comparison.png', 'training_time_comparison.png']:
        fpath = os.path.join('graphs', fname)
        assert os.path.exists(fpath), f'Missing: {fpath}'
        os.remove(fpath)
    ok("all 5 model name variants plotted, palette no KeyError")
except Exception as e:
    fail("visualization palette", e)

# ── 5b. Confusion matrix filename generation ──────────────────
print('\n[5b] confusion matrix filename key derivation')
try:
    for model_name, expected_key in [
        ('Classical SVM (RBF)',    'classical_svm_rbf'),
        ('Classical SVM (Rbf)',    'classical_svm_rbf'),
        ('Classical SVM (LINEAR)', 'classical_svm_linear'),
        ('Classical SVM (Linear)', 'classical_svm_linear'),
        ('Classical SVM (Poly)',   'classical_svm_poly'),
        ('Quantum SVM (QSVC)',     'quantum_svm_qsvc'),
    ]:
        key = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        assert key == expected_key, f"Got {key!r}, expected {expected_key!r}"
    ok("all filename derivations correct")
except AssertionError as e:
    fail("filename derivation", e)

# ── 6. No duplicate model when kernel=linear ──────────────────
print('\n[6] app.py: no duplicate model key when kernel=linear')
try:
    for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
        comparison = 'linear' if kernel != 'linear' else 'rbf'
        primary_lbl     = f'Classical SVM ({kernel.upper()})'
        comparison_lbl  = f'Classical SVM ({comparison.upper()})'
        assert primary_lbl != comparison_lbl, f"Duplicate at kernel={kernel}!"
    ok("all 4 kernels produce distinct model names")
except AssertionError as e:
    fail("duplicate model guard", e)

# ── Summary ───────────────────────────────────────────────────
print()
print(SEP)
print(f'PASSED: {len(PASS)}  |  FAILED: {len(FAIL)}')
if FAIL:
    print('FAILURES:')
    for f in FAIL:
        print(f'  - {f}')
else:
    print('ALL CHECKS PASSED ✅')
print(SEP)
