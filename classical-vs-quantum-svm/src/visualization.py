import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set premium styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Outfit', 'Inter', 'Roboto', 'DejaVu Sans'],
    'figure.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150
})

def plot_performance_metrics(df, output_dir="graphs"):
    """
    Generates bar charts comparing Accuracy, F1 Score, and Training Time 
    across datasets and models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing columns: Dataset, Model, Accuracy, Precision, Recall, F1_Score, Train_Time_Sec, Predict_Time_Sec
    output_dir : str
        Directory to save the generated graphs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build palette dynamically from actual model names in the dataframe
    # so it works regardless of kernel name capitalisation (Rbf vs RBF etc.)
    _base_colors = ['#2b5c8f', '#f27c38', '#20b2aa', '#9b59b6', '#e74c3c', '#27ae60']
    unique_models = df["Model"].unique().tolist()
    colors = {}
    for i, m in enumerate(unique_models):
        m_lower = m.lower()
        if 'quantum' in m_lower:
            colors[m] = '#20b2aa'
        elif 'linear' in m_lower:
            colors[m] = '#f27c38'
        else:
            colors[m] = _base_colors[i % len(_base_colors)]

    # 1. Compare Accuracy
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Dataset", y="Accuracy", hue="Model", data=df, palette=colors)
    plt.title("Model Accuracy Comparison", pad=20, weight='bold')
    plt.ylabel("Accuracy Score (0.0 - 1.0)")
    plt.xlabel("Dataset")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Compare F1 Score
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Dataset", y="F1_Score", hue="Model", data=df, palette=colors)
    plt.title("Model F1 Score Comparison", pad=20, weight='bold')
    plt.ylabel("F1 Score (weighted)")
    plt.xlabel("Dataset")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_comparison.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Compare Training Time (Log Scale is essential due to orders of magnitude differences)
    plt.figure(figsize=(10, 6))
    # Add a small offset to prevent log(0) issues if training is extremely fast
    df_copy = df.copy()
    df_copy['Train_Time_Sec_Offset'] = df_copy['Train_Time_Sec'] + 1e-6
    ax = sns.barplot(x="Dataset", y="Train_Time_Sec_Offset", hue="Model", data=df_copy, palette=colors)
    plt.yscale('log')
    plt.title("Model Training Time Comparison (Log Scale)", pad=20, weight='bold')
    plt.ylabel("Training Time (Seconds, log scale)")
    plt.xlabel("Dataset")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Add exact time values on top of bars
    for container in ax.containers:
        # We retrieve the actual unoffsetted values for labels
        labels = []
        for bar in container:
            val = bar.get_height()
            if val > 1e-5:
                # Retrieve matching original training time
                labels.append(f"{val:.4f}s")
            else:
                labels.append("~0s")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time_comparison.png"), bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(results_dict, output_dir="graphs"):
    """
    Plots confusion matrix heatmaps for each run.
    
    Parameters:
    -----------
    results_dict : dict
        A dictionary mapping (dataset_name, model_name) to evaluation metrics.
    output_dir : str
        Directory to save confusion matrices.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for key, metrics in results_dict.items():
        dataset_name, model_name = key
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(6, 5))
        # Custom gradient color map
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=True, yticklabels=True,
                    annot_kws={"size": 12, "weight": "bold"})
        
        plt.title(f"Confusion Matrix: {model_name}\n({dataset_name} Dataset)", pad=15, weight='bold')
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        
        plt.savefig(confusion_matrix_path(dataset_name, model_name, output_dir), dpi=150)
        plt.close()


def confusion_matrix_path(dataset_name, model_name, output_dir="graphs"):
    """Return the saved confusion-matrix image path for a dataset/model pair."""
    clean_model_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    clean_dataset_name = dataset_name.lower().replace(" ", "_")
    filename = f"confusion_matrix_{clean_dataset_name}_{clean_model_name}.png"
    return os.path.join(output_dir, filename)
