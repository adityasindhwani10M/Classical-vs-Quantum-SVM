import time
from sklearn.svm import SVC

def train_classical_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale', random_state=42):
    """
    Trains a classical Support Vector Machine (SVC) and measures training time.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training labels.
    kernel : str
        The kernel type to be used in the algorithm ('rbf', 'linear', 'poly').
    C : float
        Regularization parameter.
    gamma : str or float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    model : sklearn.svm.SVC
        The trained SVM model.
    train_time : float
        Time taken to train the model in seconds.
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
    
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    end_time = time.perf_counter()
    
    train_time = end_time - start_time
    return model, train_time

def predict_classical_svm(model, X_test):
    """
    Makes predictions using the trained classical SVM model and measures prediction time.
    
    Parameters:
    -----------
    model : sklearn.svm.SVC
        Trained SVM model.
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
