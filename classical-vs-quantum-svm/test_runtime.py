import sys, io, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Create a synthetic CSV for testing
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target_names[iris.target]
csv_buf = io.BytesIO(df.to_csv(index=False).encode('utf-8'))
csv_buf.name = 'iris_test.csv'

# 1. Load
from src.dataset_loader import load_uploaded_dataset
X, y, feature_names, target_names, dataset_name = load_uploaded_dataset(csv_buf, target_column='target', dataset_name='iris_test.csv')
print(f'Load OK: X={X.shape}, y={y.shape}, classes={target_names}, name={dataset_name}')

# 2. Preprocess
from src.preprocessing import preprocess_dataset
X_train, X_test, y_train, y_test = preprocess_dataset(X, y, test_size=0.2, max_samples=50, n_components=2, random_state=42)
print(f'Preprocess OK: train={X_train.shape}, test={X_test.shape}')

# 3. Classical SVM
from src.classical_svm import train_classical_svm, predict_classical_svm
model, t = train_classical_svm(X_train, y_train, kernel='rbf')
preds, pt = predict_classical_svm(model, X_test)
print(f'Classical SVM OK: train_time={t:.4f}s, preds={preds[:5]}')

# 4. Evaluation
from src.evaluation import evaluate_predictions, save_results_to_csv
metrics = evaluate_predictions(y_test, preds, t, pt)
print(f'Evaluation OK: accuracy={metrics["accuracy"]:.4f}, f1={metrics["f1_score"]:.4f}')

# 5. Visualization - needs at least 2 rows for some charts, add a second dummy model
from src.visualization import plot_performance_metrics, plot_confusion_matrices
results = {('iris', 'Classical SVM (RBF)'): metrics}
os.makedirs('graphs', exist_ok=True)
rows = [
    {'Dataset': 'iris', 'Model': 'Classical SVM (RBF)', 'Accuracy': metrics['accuracy'],
     'Precision': metrics['precision'], 'Recall': metrics['recall'],
     'F1_Score': metrics['f1_score'], 'Train_Time_Sec': metrics['train_time'],
     'Predict_Time_Sec': metrics['predict_time']}
]
df_res = pd.DataFrame(rows)
plot_performance_metrics(df_res, 'graphs')
plot_confusion_matrices(results, 'graphs')
print('Visualization OK: graphs generated')

# 6. Test Quantum SVM (small subset)
print('\nTesting Quantum SVM (small subset)...')
from src.quantum_svm import train_quantum_svm, predict_quantum_svm
# Use a very small binary subset for speed
binary_mask = y_train < 2
X_q_train = X_train[binary_mask][:10]
y_q_train = y_train[binary_mask][:10]
binary_mask_test = y_test < 2
X_q_test = X_test[binary_mask_test][:5]
y_q_test = y_test[binary_mask_test][:5]

q_model, q_t = train_quantum_svm(X_q_train, y_q_train, reps=1, entanglement='linear')
q_preds, q_pt = predict_quantum_svm(q_model, X_q_test)
q_metrics = evaluate_predictions(y_q_test, q_preds, q_t, q_pt)
print(f'Quantum SVM OK: train_time={q_t:.4f}s, accuracy={q_metrics["accuracy"]:.4f}')

# 7. save_results_to_csv
os.makedirs('results', exist_ok=True)
save_results_to_csv({('iris', 'Classical SVM (RBF)'): metrics}, 'results/test_comparison_results.csv')
print('CSV save OK')

print()
print('=== ALL TESTS PASSED ===')
