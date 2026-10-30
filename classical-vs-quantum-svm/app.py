import os
import time
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Import our custom modules
from src.dataset_loader import load_uploaded_dataset, get_csv_column_names, read_csv_preview
from src.preprocessing import preprocess_dataset
from src.classical_svm import train_classical_svm, predict_classical_svm
from src.quantum_svm import train_quantum_svm, predict_quantum_svm
from src.evaluation import evaluate_predictions, save_results_to_csv
from src.visualization import plot_performance_metrics, plot_confusion_matrices, confusion_matrix_path

# Set page configurations
st.set_page_config(
    page_title="Quantum SVM Study Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    font-family: 'Inter', sans-serif;
    overflow-y: auto !important;
    overflow-x: hidden;
    height: auto !important;
    min-height: 100vh;
}

.main-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #64748b;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1),
                0 2px 4px -2px rgb(0 0 0 / 0.1);
    border: 1px solid #e2e8f0;
    text-align: center;
}

.metric-val {
    font-size: 2.2rem;
    font-weight: 700;
    color: #4f46e5;
    font-family: 'Outfit', sans-serif;
}

.metric-label {
    font-size: .875rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: #64748b;
    font-weight: 600;
    margin-top: .5rem;
}

.section-header {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: #1e293b;
    border-left: 5px solid #4f46e5;
    padding-left: 10px;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Main Title & Description
st.markdown('<div class="main-title">⚛️ Classical vs. Quantum SVM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A comparative research study evaluating Support Vector Machines with Fidelity Quantum Kernels on NISQ simulators.</div>', unsafe_allow_html=True)

# Dataset upload
st.markdown('<div class="section-header">📂 Upload Datasets</div>', unsafe_allow_html=True)
st.markdown(
    "Upload one or more **CSV** files to compare Classical and Quantum SVM on your own data. "
    "Both numeric and categorical/non-numeric feature columns are supported — "
    "categorical columns are automatically label-encoded. Header-less CSVs are also auto-detected."
)

uploaded_files = st.file_uploader(
    "Choose CSV dataset files",
    type=["csv", "txt"],
    accept_multiple_files=True,
    help="You can upload multiple CSV or TXT files. Each file is evaluated separately.",
)

target_column = None
if uploaded_files:
    try:
        column_options = get_csv_column_names(uploaded_files[0])
        uploaded_files[0].seek(0)
    except Exception as e:
        st.error(f"❌ Failed to parse columns from **{uploaded_files[0].name}**: {e}")
        column_options = []
        
    selectbox_index = len(column_options) - 1 if column_options else None
    target_column = st.selectbox(
        "Target column (applied to all uploaded files)",
        options=column_options,
        index=selectbox_index,
        help="Select which column contains the class labels. If not found in a file, the last column is used automatically.",
    )

    st.markdown("#### Uploaded files")
    for uploaded_file in uploaded_files:
        with st.expander(f"Preview: {uploaded_file.name}", expanded=len(uploaded_files) == 1):
            st.dataframe(read_csv_preview(uploaded_file, nrows=10), width="stretch")
            uploaded_file.seek(0)
else:
    st.info("No datasets uploaded yet. Add one or more CSV files above to begin.")

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Simulation Settings")

st.sidebar.markdown("### 📊 Hyperparameters")

max_load_rows = st.sidebar.number_input(
    "Max Rows to Load from CSV",
    min_value=100,
    max_value=1000000,
    value=10000,
    step=1000,
    help="Limits the number of rows loaded from large CSV files to prevent memory issues and slow parsing."
)

run_quantum = st.sidebar.checkbox(
    "Run Quantum SVM Simulation",
    value=True,
    help="Uncheck to skip Quantum SVM and run Classical SVM on much larger datasets without timeouts."
)

if run_quantum:
    max_samples = st.sidebar.slider(
        "Max Dataset Samples",
        min_value=20,
        max_value=300,
        value=100,
        step=10,
        help="Size to downsample the dataset to keep the local simulation speed reasonable (capped at 300 when Quantum is enabled)."
    )
else:
    max_samples = st.sidebar.slider(
        "Max Dataset Samples",
        min_value=20,
        max_value=5000,
        value=1000,
        step=100,
        help="Size to downsample the dataset. Can be higher since Quantum SVM is skipped."
    )

pca_components = st.sidebar.slider(
    "PCA Components (Qubits)",
    min_value=2,
    max_value=4,
    value=2,
    step=1,
    help="Number of dimensions features are reduced to. Maps directly to the number of qubits."
)

test_size = st.sidebar.slider(
    "Test Set Split Ratio",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="Fraction of data reserved for evaluation (e.g., 0.20 = 20%)."
)

st.sidebar.markdown("### 🧬 SVM Kernel Configuration")
classical_kernel = st.sidebar.selectbox(
    "Classical SVM Kernel",
    options=["rbf", "linear", "poly", "sigmoid"],
    index=0
)

if run_quantum:
    quantum_reps = st.sidebar.slider(
        "ZZFeatureMap Repetitions",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Number of repeating quantum circuit blocks in the feature map."
    )

    quantum_entanglement = st.sidebar.selectbox(
        "ZZFeatureMap Entanglement",
        options=["linear", "circular", "full"],
        index=0,
        help="Entanglement pattern for features mapped to qubits."
    )
else:
    quantum_reps = 2
    quantum_entanglement = "linear"

run_button = st.sidebar.button("Run Comparative Study 🚀", use_container_width=True)

# Define file paths
csv_path = os.path.join("results", "comparison_results.csv")
report_path = os.path.join("report", "comparison_report.md")

# Label for the selected classical SVM kernel
primary_label = f"Classical SVM ({classical_kernel.upper()})"

# Helper function to generate Markdown report file
def generate_markdown_report_file(df, results_dict):
    os.makedirs("report", exist_ok=True)
    
    md_table = "| Dataset | Model | Accuracy | Precision | Recall | F1 Score | Train Time (s) | Predict Time (s) |\n"
    md_table += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for _, r in df.iterrows():
        md_table += (
            f"| {r['Dataset'].capitalize()} | {r['Model']} | {r['Accuracy']:.4f} | "
            f"{r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1_Score']:.4f} | "
            f"{r['Train_Time_Sec']:.4f} | {r['Predict_Time_Sec']:.4f} |\n"
        )
        
    observations_section = ""
    for dataset in df['Dataset'].unique():
        models = df[df['Dataset'] == dataset]
        observations_section += f"### {dataset.capitalize()} Dataset Analysis\n"
        if not models.empty:
            best_model = models.loc[models['Accuracy'].idxmax()]['Model']
            best_acc = models['Accuracy'].max()
            observations_section += f"- **Best Performing Model**: {best_model} with an accuracy of **{best_acc:.2%}**.\n"
            
            q_row = models[models['Model'].str.contains('Quantum')]
            # FIX: Search for any Classical model (not just 'RBF'), pick the first non-linear one
            c_classical_row = models[models['Model'].str.contains('Classical')]
            if not q_row.empty and not c_classical_row.empty:
                q_time = q_row.iloc[0]['Train_Time_Sec']
                c_time = c_classical_row.iloc[0]['Train_Time_Sec']
                c_name = c_classical_row.iloc[0]['Model']
                slowdown = q_time / c_time if c_time > 0 else float('inf')
                observations_section += (
                    f"- **Computational Cost**: The Quantum SVM took **{q_time:.4f}s** to train, "
                    f"compared to **{c_time:.4f}s** for {c_name}, "
                    f"representing a **{slowdown:.1f}x slowdown**.\n"
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
   Classical SVM ({classical_kernel.upper()}) achieved competitive or superior performance on these low-dimensional datasets. The Quantum SVM (QSVC) performs comparably to the classical models, indicating that the quantum kernel successfully maps the data to a high-dimensional Hilbert space where the data remains separable.
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
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

# Initialize Session State to hold results from the current session
if "df" not in st.session_state:
    st.session_state.df = None

# Running the simulation
if run_button:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one CSV dataset before running the study.")
    else:
        st.info("🚀 Initiating comparative study. Please do not close the browser...")
        
        # UI status block
        status_container = st.container()
        progress_bar = st.progress(0.0)
        
        results = {}
        total_steps = len(uploaded_files)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            step_pct = float(idx) / total_steps
            progress_bar.progress(step_pct)
            
            with status_container:
                try:
                    X, y, feature_names, target_names, dataset_name = load_uploaded_dataset(
                        uploaded_file,
                        target_column=target_column,
                        dataset_name=uploaded_file.name,
                        max_rows=max_load_rows,
                    )
                except Exception as e:
                    st.error(f"❌ Failed to load **{uploaded_file.name}**: {e}")
                    uploaded_file.seek(0)
                    continue

                st.write(f"### ⚙️ Processing **{dataset_name.upper()}** ({uploaded_file.name})...")
                
                # 1. Load Data
                load_msg = st.empty()
                load_msg.write(
                    f"✅ Loaded {len(y)} samples with {X.shape[1]} features and {len(target_names)} classes."
                )
                uploaded_file.seek(0)
                
                # 2. Preprocess Data
                prep_msg = st.empty()
                prep_msg.write("⏳ Preprocessing and performing PCA reduction...")
                X_train, X_test, y_train, y_test = preprocess_dataset(
                    X, y, test_size=test_size, max_samples=max_samples, n_components=pca_components, random_state=42
                )
                prep_msg.write(
                    f"✅ Preprocessed: Train size={X_train.shape[0]} | Test size={X_test.shape[0]}. "
                    f"PCA complete (components={X_train.shape[1]}, qubits={X_train.shape[1]})."
                )
                
                # --- MODEL 1: User's chosen Classical SVM kernel ---
                c_rbf_msg = st.empty()
                c_rbf_msg.write(f"⏳ Training {primary_label}...")
                try:
                    c_rbf_model, c_rbf_train_time = train_classical_svm(X_train, y_train, kernel=classical_kernel)
                    c_rbf_preds, c_rbf_pred_time = predict_classical_svm(c_rbf_model, X_test)
                    c_rbf_metrics = evaluate_predictions(y_test, c_rbf_preds, c_rbf_train_time, c_rbf_pred_time)
                    results[(dataset_name, primary_label)] = c_rbf_metrics
                    c_rbf_msg.write(f"✅ {primary_label} complete: Acc = **{c_rbf_metrics['accuracy']:.4f}**")
                except Exception as e:
                    c_rbf_msg.error(f"❌ {primary_label} failed: {e}")
                
                # # --- MODEL 2: Comparison Classical SVM (always different from Model 1) ---
                # c_lin_msg = st.empty()
                # c_lin_msg.write(f"⏳ Training {comparison_label}...")
                # try:
                #     c_lin_model, c_lin_train_time = train_classical_svm(X_train, y_train, kernel=comparison_kernel)
                #     c_lin_preds, c_lin_pred_time = predict_classical_svm(c_lin_model, X_test)
                #     c_lin_metrics = evaluate_predictions(y_test, c_lin_preds, c_lin_train_time, c_lin_pred_time)
                #     results[(dataset_name, comparison_label)] = c_lin_metrics
                #     c_lin_msg.write(f"✅ {comparison_label} complete: Acc = **{c_lin_metrics['accuracy']:.4f}**")
                # except Exception as e:
                #     c_lin_msg.error(f"❌ {comparison_label} failed: {e}")
                
                # --- MODEL 3: Quantum SVM (QSVC) ---
                if run_quantum:
                    q_msg = st.empty()
                    q_msg.write("⏳ Simulating Quantum SVM (QSVC + ZZFeatureMap)... This may take 30s–2min...")
                    try:
                        q_model, q_train_time = train_quantum_svm(X_train, y_train, reps=quantum_reps, entanglement=quantum_entanglement)
                        q_preds, q_pred_time = predict_quantum_svm(q_model, X_test)
                        q_metrics = evaluate_predictions(y_test, q_preds, q_train_time, q_pred_time)
                        results[(dataset_name, 'Quantum SVM (QSVC)')] = q_metrics
                        q_msg.write(f"✅ Quantum SVM complete: Acc = **{q_metrics['accuracy']:.4f}** | Time = **{q_train_time:.2f}s**")
                    except Exception as e:
                        q_msg.error(f"❌ Quantum SVM failed: {e}")
                    
            if idx < len(uploaded_files) - 1:
                st.divider()

        progress_bar.progress(1.0)
        status_container.empty()
            
        if not results:
            st.error("No datasets were processed successfully. Please check your CSV files and try again.")
            st.stop()
        
        # Save to files
        save_results_to_csv(results, csv_path)
        
        # Build Dataframe
        rows = []
        for (dataset, model), metrics in results.items():
            rows.append({
                'Dataset': dataset,
                'Model': model,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'Train_Time_Sec': metrics['train_time'],
                'Predict_Time_Sec': metrics['predict_time']
            })
        df = pd.DataFrame(rows)
        st.session_state.df = df
        
        # Generate graphs
        plot_performance_metrics(df, "graphs")
        plot_confusion_matrices(results, "graphs")
        
        # Generate markdown report
        generate_markdown_report_file(df, results)
        
        st.success("🎉 Experiments completed successfully! Results updated below.")

# Display content if results are available
if st.session_state.df is not None:
    df = st.session_state.df
    
    # 1. Metric Cards
    st.markdown('<div class="section-header">🏆 High-Level Performance Metrics</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    # FIX: Defensive guards against empty filtered dataframes
    classical_df = df[df['Model'].str.contains('Classical')]
    quantum_df   = df[df['Model'].str.contains('Quantum')]

    if not classical_df.empty:
        best_classical = classical_df.loc[classical_df['Accuracy'].idxmax()]
        best_c_acc = f"{best_classical['Accuracy']:.1%}"
        best_c_label = str(best_classical['Dataset']).capitalize()
    else:
        best_c_acc   = "N/A"
        best_c_label = "N/A"

    if not quantum_df.empty:
        best_quantum = quantum_df.loc[quantum_df['Accuracy'].idxmax()]
        best_q_acc   = f"{best_quantum['Accuracy']:.1%}"
        best_q_label = str(best_quantum['Dataset']).capitalize()
        avg_q_time   = quantum_df['Train_Time_Sec'].mean()
        avg_c_time   = classical_df['Train_Time_Sec'].mean() if not classical_df.empty else 1.0
        slowdown     = avg_q_time / avg_c_time if avg_c_time > 0 else 0.0
    else:
        best_q_acc   = "N/A"
        best_q_label = "N/A"
        slowdown     = 0.0
        
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{best_c_acc}</div>
                <div class="metric-label">Best Classical Accuracy ({best_c_label})</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{best_q_acc}</div>
                <div class="metric-label">Best Quantum Accuracy ({best_q_label})</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c3:
        slowdown_text = f"{slowdown:.1f}x" if not quantum_df.empty else "N/A"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{slowdown_text}</div>
                <div class="metric-label">Average Quantum Simulator Slowdown</div>
            </div>
        """, unsafe_allow_html=True)
        
    # Tabs layout
    st.markdown('<div class="section-header">🔍 In-depth Analysis Tabs</div>', unsafe_allow_html=True)
    tab_dashboard, tab_cf, tab_report = st.tabs(["📊 Performance Dashboard", "🔬 Confusion Matrices", "📄 Research Report"])
    
    with tab_dashboard:
        col_table, col_download = st.columns([3, 1])
        with col_table:
            st.write("### Comparison Summary Table")
            st.dataframe(
                df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1_Score': '{:.4f}',
                    'Train_Time_Sec': '{:.4f}s',
                    'Predict_Time_Sec': '{:.4f}s'
                }),
                width="stretch"
            )
        with col_download:
            st.write("### Data Export")
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download results.csv",
                data=csv_data,
                file_name="svm_comparison_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        st.write("---")
        
        # Charts Columns
        col_acc, col_f1 = st.columns(2)
        with col_acc:
            st.write("### Accuracy Comparison")
            if os.path.exists("graphs/accuracy_comparison.png"):
                st.image("graphs/accuracy_comparison.png", width="stretch")
            else:
                st.info("Run the simulation to generate the accuracy comparison chart.")
                
        with col_f1:
            st.write("### F1-Score Comparison")
            if os.path.exists("graphs/f1_score_comparison.png"):
                st.image("graphs/f1_score_comparison.png", width="stretch")
            else:
                st.info("Run the simulation to generate the F1-score comparison chart.")
                
        st.write("---")
        st.write("### Computational Speed Comparison")
        if os.path.exists("graphs/training_time_comparison.png"):
            st.image("graphs/training_time_comparison.png", width="stretch")
        else:
            st.info("Run the simulation to generate the training time comparison chart.")
            
    with tab_cf:
        st.write("### Model Confusion Matrices")
        dataset_cf = st.selectbox("Select Dataset to view Confusion Matrices:", options=df['Dataset'].unique())
        
        # Display confusion matrix plots for selected dataset
        st.write(f"Confusion Matrices for **{dataset_cf.upper()}** dataset:")
        
        dataset_models = df[df["Dataset"] == dataset_cf]["Model"].tolist()
        classical_models = [m for m in dataset_models if "Classical" in m]
        quantum_models = [m for m in dataset_models if "Quantum" in m]
        models_to_show = dataset_models

        if not models_to_show:
            st.info("No confusion matrices available for this dataset.")
        else:
            cols = st.columns(len(models_to_show))
            for col, model_name in zip(cols, models_to_show):
                cf_file = confusion_matrix_path(dataset_cf, model_name)
                with col:
                    st.write(f"**{model_name}**")
                    if os.path.exists(cf_file):
                        st.image(cf_file, width="stretch")
                    else:
                        st.info("Run simulation to generate this matrix.")
                
    with tab_report:
        st.write("### Research Comparison Report")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_md = f.read()
            st.markdown(report_md)
        else:
            st.info("Markdown report not found. Please click 'Run Comparative Study' in the sidebar to generate it.")
else:
    st.warning(
        "📊 No results found yet. Upload one or more CSV datasets, configure options in the sidebar, "
        "and click **Run Comparative Study** to generate metrics!"
    )