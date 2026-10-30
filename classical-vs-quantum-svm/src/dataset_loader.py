import io
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

__all__ = ["load_uploaded_dataset", "get_csv_column_names", "read_csv_preview"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_dataset_name(name):
    """Convert a filename into a safe dataset identifier."""
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r"[^\w\-]", "_", name)
    return name.lower() or "dataset"


def _has_header(file_obj):
    """
    Heuristically detect whether a CSV file has a header row.

    Strategy: Read the first row and the rest of the file.
    If the first row contains values that look like labels/strings and
    the same columns in subsequent rows are numeric, then the first row
    is a header. Otherwise treat the file as header-less.
    """
    try:
        if isinstance(file_obj, (str, os.PathLike)):
            with open(file_obj, "rb") as f:
                sample = f.read(8192)
        else:
            sample = file_obj.read(8192)
            file_obj.seek(0)
    except Exception:
        return True  # default: assume header exists

    if isinstance(sample, bytes):
        sample = sample.decode("utf-8", errors="replace")

    lines = [l for l in sample.splitlines() if l.strip()]
    if len(lines) < 2:
        return True  # can't determine, assume header

    first_row = lines[0].split(",")
    second_row = lines[1].split(",")

    if len(first_row) != len(second_row):
        return True

    # Count how many fields in first row are non-numeric strings
    first_non_numeric = 0
    for val in first_row:
        val = val.strip().strip('"').strip("'")
        try:
            float(val)
        except ValueError:
            first_non_numeric += 1

    # Count how many fields in second row are non-numeric strings
    second_non_numeric = 0
    for val in second_row:
        val = val.strip().strip('"').strip("'")
        try:
            float(val)
        except ValueError:
            second_non_numeric += 1

    # If first row has MORE non-numeric fields than the second row,
    # it is almost certainly a header row of column names.
    return first_non_numeric > second_non_numeric


def _read_raw(file_obj, nrows=None):
    """
    Internal: read a CSV using header auto-detection.
    Always seeks back to position 0 before returning.
    Returns a DataFrame with clean string column names.
    """
    # 1. Determine header presence
    has_hdr = _has_header(file_obj)

    # 2. Reset pointer if seekable
    if hasattr(file_obj, 'seek'):
        try:
            file_obj.seek(0)
        except Exception:
            pass

    def parse_csv(enc):
        if hasattr(file_obj, 'seek'):
            try:
                file_obj.seek(0)
            except Exception:
                pass
        
        target = file_obj
        if has_hdr:
            return pd.read_csv(target, nrows=nrows, encoding=enc, on_bad_lines='skip')
        else:
            df = pd.read_csv(target, header=None, nrows=nrows, encoding=enc, on_bad_lines='skip')
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
            return df

    try:
        df = parse_csv('utf-8')
    except (UnicodeDecodeError, ValueError):
        try:
            df = parse_csv('latin-1')
        except (UnicodeDecodeError, ValueError):
            try:
                df = parse_csv('cp1252')
            except Exception as e:
                raise ValueError(f"Could not parse CSV file. Encoding or parsing error: {e}")
        except Exception as e:
            raise ValueError(f"Could not parse CSV file: {e}")
    except Exception as e:
        raise ValueError(f"Could not parse CSV file: {e}")

    # Rewind so the caller can read again if needed
    if hasattr(file_obj, 'seek'):
        try:
            file_obj.seek(0)
        except Exception:
            pass

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_csv_column_names(file_obj):
    """
    Return the list of column names from a CSV file without loading all data.

    Parameters
    ----------
    file_obj : file-like object
        Uploaded CSV file (e.g. from st.file_uploader).

    Returns
    -------
    columns : list[str]
        Column names. Falls back to positional names (col_0, col_1, …) for
        header-less files.
    """
    df = _read_raw(file_obj, nrows=0)   # header-only read — very fast
    return list(df.columns)


def read_csv_preview(file_obj, nrows=10):
    """
    Return the first *nrows* rows of a CSV as a DataFrame for display.

    Parameters
    ----------
    file_obj : file-like object
        Uploaded CSV file.
    nrows : int
        Number of rows to return (default 10).

    Returns
    -------
    preview_df : pandas.DataFrame
    """
    return _read_raw(file_obj, nrows=nrows)


def load_uploaded_dataset(file_obj, target_column=None, dataset_name=None, max_rows=None):
    """
    Load a classification dataset from an uploaded CSV file.

    Supports:
    - CSV files with or without a header row (auto-detected)
    - Non-numeric (categorical) feature columns (auto label-encoded)
    - Missing values in features (imputed) and target (rows dropped)
    - Fallback to last column when specified target column is not found

    Parameters
    ----------
    file_obj : file-like or str (path)
        Uploaded CSV file object (e.g. from st.file_uploader) or a path string.
    target_column : str or None
        Name/index of the label column. Defaults to the last column.
    dataset_name : str or None
        Display name for the dataset. Defaults to the uploaded filename.
    max_rows : int or None
        Maximum number of rows to load from the CSV file.

    Returns
    -------
    X : numpy.ndarray            Feature matrix (all numeric after encoding).
    y : numpy.ndarray            Encoded integer target labels.
    feature_names : list[str]   Names of the feature columns.
    target_names : list[str]    Names of the target classes.
    dataset_name : str           Sanitized dataset name.
    """
    df = _read_raw(file_obj, nrows=max_rows)

    if df.empty:
        raise ValueError("Uploaded file is empty.")

    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least one feature column and one target column.")

    # --- Target column resolution ---
    if target_column is not None and target_column in df.columns:
        pass  # exact match found
    else:
        target_column = df.columns[-1]  # fall back to last column

    # --- Drop rows where target is null ---
    if df[target_column].isnull().any():
        df = df.dropna(subset=[target_column])
        if df.empty:
            raise ValueError("All rows in target column are null/missing after dropping nulls.")

    feature_cols = [col for col in df.columns if col != target_column]
    if not feature_cols:
        raise ValueError("No feature columns found after selecting the target column.")

    feature_df = df[feature_cols].copy()

    # --- Encode features: numeric → impute NaN; categorical → LabelEncode ---
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            if feature_df[col].isnull().any():
                median_val = feature_df[col].median()
                feature_df[col] = feature_df[col].fillna(0.0 if pd.isnull(median_val) else median_val)
        else:
            feature_df[col] = feature_df[col].fillna("missing_value").astype(str)
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col])

    X = feature_df.to_numpy(dtype=float)
    y_raw = df[target_column].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    target_names = [str(lbl) for lbl in label_encoder.classes_]

    if len(target_names) < 2:
        raise ValueError(
            f"The resolved target column '{target_column}' contains only 1 unique class in the loaded data: {target_names}. "
            "Classification requires at least 2 unique classes. Please choose a different target column (e.g. from the selectbox) "
            "or load more rows from the CSV/text file."
        )

    if dataset_name is None:
        dataset_name = getattr(file_obj, "name", "uploaded_dataset")

    return X, y, list(feature_cols), target_names, _sanitize_dataset_name(dataset_name)
