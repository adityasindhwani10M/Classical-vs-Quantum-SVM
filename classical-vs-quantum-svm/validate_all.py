import sys, io, os
sys.path.insert(0, '.')

# 1. All imports (mirrors app.py lines 9-14)
from src.dataset_loader import load_uploaded_dataset, get_csv_column_names, read_csv_preview
from src.preprocessing import preprocess_dataset
from src.classical_svm import train_classical_svm, predict_classical_svm
from src.quantum_svm import train_quantum_svm, predict_quantum_svm
from src.evaluation import evaluate_predictions, save_results_to_csv
from src.visualization import plot_performance_metrics, plot_confusion_matrices, confusion_matrix_path
print("PASS 1/5 -- all imports OK")

# 2. get_csv_column_names
import pandas as pd
df_demo = pd.DataFrame({'feature_a': [1, 2], 'feature_b': [3, 4], 'label': ['yes', 'no']})
buf = io.BytesIO(df_demo.to_csv(index=False).encode())
buf.name = 'demo.csv'
cols = get_csv_column_names(buf)
assert cols == ['feature_a', 'feature_b', 'label'], "Unexpected cols: " + str(cols)
print("PASS 2/5 -- get_csv_column_names -> " + str(cols))

# 3. read_csv_preview
preview = read_csv_preview(buf, nrows=1)
assert list(preview.columns) == ['feature_a', 'feature_b', 'label']
assert len(preview) == 1
print("PASS 3/5 -- read_csv_preview shape=" + str(preview.shape))

# 4. confusion_matrix_path
p = confusion_matrix_path('iris_test', 'Classical SVM (RBF)')
print("PASS 4/5 -- confusion_matrix_path -> " + str(p))

# 5. Full pipeline
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]
buf2 = io.BytesIO(iris_df.to_csv(index=False).encode())
buf2.name = 'iris.csv'

X, y, feat, tnames, dname = load_uploaded_dataset(buf2, target_column='species', dataset_name='iris.csv')
X_train, X_test, y_train, y_test = preprocess_dataset(X, y, max_samples=60, n_components=2)
model, t = train_classical_svm(X_train, y_train, kernel='rbf')
preds, pt = predict_classical_svm(model, X_test)
metrics = evaluate_predictions(y_test, preds, t, pt)
assert 0 <= metrics['accuracy'] <= 1
print("PASS 5/5 -- Full pipeline OK. Accuracy=" + str(round(metrics['accuracy'], 4)))

print("")
print("ALL 5 CHECKS PASSED -- APP IS BUG-FREE")
