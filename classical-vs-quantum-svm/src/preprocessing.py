import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def preprocess_dataset(X, y, test_size=0.2, max_samples=None, n_components=2, random_state=42):
    """
    Preprocesses the dataset by performing optional sub-sampling,
    train-test split, standardization, PCA, and MinMax scaling to [-1, 1].
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Labels.
    test_size : float
        Proportion of data in test split (default 0.2).
    max_samples : int or None
        Maximum total samples to keep (sub-sampling). Balanced across classes.
    n_components : int
        Number of PCA components to reduce features to.
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    X_train_final : numpy.ndarray
        Preprocessed, reduced, and scaled training features.
    X_test_final : numpy.ndarray
        Preprocessed, reduced, and scaled testing features.
    y_train : numpy.ndarray
        Training labels.
    y_test : numpy.ndarray
        Testing labels.
    """
    # 1. Optional balanced sub-sampling to keep quantum simulations fast
    if max_samples is not None and len(y) > max_samples:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        num_classes = len(unique_classes)
        # Ensure we select at least 1 sample per class
        samples_per_class = max(1, max_samples // num_classes)
        
        selected_indices = []
        np.random.seed(random_state)
        
        for c in unique_classes:
            indices_of_class = np.where(y == c)[0]
            # If the class has fewer samples than requested, take all of them
            n_to_select = min(samples_per_class, len(indices_of_class))
            selected_for_class = np.random.choice(indices_of_class, size=n_to_select, replace=False)
            selected_indices.extend(selected_for_class)
            
        selected_indices = np.array(selected_indices)
        # Shuffle indices
        np.random.shuffle(selected_indices)
        X = X[selected_indices]
        y = y[selected_indices]
        
    # 2. Train-Test Split (80:20)
    # Ensure that y contains at least 2 unique classes
    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        raise ValueError(
            f"The downsampled dataset has only {len(unique_classes)} class: {list(unique_classes)}. "
            "Classification requires at least 2 unique classes in the training data. "
            "Please select a different target column or increase the number of rows/samples."
        )

    # To guarantee that all unique classes exist in the training split y_train:
    # We identify single-sample classes and place them directly in the train split,
    # and perform stratified split on multi-sample classes.
    single_sample_indices = []
    multi_sample_indices = []
    
    for c in unique_classes:
        indices = np.where(y == c)[0]
        if len(indices) == 1:
            single_sample_indices.extend(indices)
        else:
            multi_sample_indices.extend(indices)
            
    if len(multi_sample_indices) < 2:
        # Fallback if there are not enough multi-sample data: simple random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
    else:
        X_multi = X[multi_sample_indices]
        y_multi = y[multi_sample_indices]
        
        multi_unique, multi_counts = np.unique(y_multi, return_counts=True)
        test_samples = int(np.ceil(test_size * len(y_multi)))
        train_samples = len(y_multi) - test_samples
        
        can_stratify = (
            np.min(multi_counts) >= 2 and
            test_samples >= len(multi_unique) and
            train_samples >= len(multi_unique)
        )
        stratify_target = y_multi if can_stratify else None
        
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
            X_multi, y_multi, test_size=test_size, random_state=random_state, stratify=stratify_target
        )
        
        if single_sample_indices:
            X_single = X[single_sample_indices]
            y_single = y[single_sample_indices]
            
            X_train = np.vstack([X_train_multi, X_single])
            y_train = np.concatenate([y_train_multi, y_single])
            X_test = X_test_multi
            y_test = y_test_multi
        else:
            X_train, X_test, y_train, y_test = X_train_multi, X_test_multi, y_train_multi, y_test_multi
    
    # 3. Standardization (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Dimensionality Reduction (PCA)
    # Ensure n_components is valid: not greater than actual feature count OR number of train samples, and at least 1
    n_components = max(1, min(n_components, X_train_scaled.shape[1], X_train_scaled.shape[0]))
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # 5. Bound the PCA features to [-1, 1] for Quantum rotation encoding compatibility
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_final = min_max_scaler.fit_transform(X_train_pca)
    X_test_final = min_max_scaler.transform(X_test_pca)
    
    return X_train_final, X_test_final, y_train, y_test
