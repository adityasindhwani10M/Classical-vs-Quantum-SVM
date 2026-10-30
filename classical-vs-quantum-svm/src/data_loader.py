"""Backward-compatible re-export. Prefer importing from dataset_loader."""

from .dataset_loader import load_uploaded_dataset

__all__ = ["load_uploaded_dataset"]
