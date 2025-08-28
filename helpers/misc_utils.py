from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re
import unicodedata
import pandas as pd
from collections.abc import Iterable
from sklearn.metrics import make_scorer, root_mean_squared_error
from mlflow.models import make_metric
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    def transform(self, y):
        return 10 * np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y / 10)

def sanitize_names(names: Iterable[str]):
    """
    Sanitize column or list of names:
    - Normalize Unicode (NFKC)
    - Replace special characters, dashes, and whitespace with underscores
    - Collapse multiple consecutive underscores
    - Strip leading/trailing underscores
    """
    sanitized = []
    for name in names:
        # Normalize hidden Unicode forms
        clean_name = unicodedata.normalize("NFKC", name)

        # Replace unwanted chars (including '-') with underscores
        clean_name = re.sub(r"[\/:.\%\"'()\[\]\s-]+", "_", clean_name)

        # Collapse multiple underscores
        clean_name = re.sub(r"_+", "_", clean_name)

        # Remove leading/trailing underscores
        clean_name = clean_name.strip("_")

        sanitized.append(clean_name)

    # Keep same type as input
    if isinstance(names, pd.Index):
        return pd.Index(sanitized, name=names.name)
    return sanitized

def rpd_score(predictions, targets):
    """RPD: Ratio of Performance to Deviation."""
    std_dev = np.std(targets, ddof=1)
    rmse = root_mean_squared_error(targets, predictions)
    return std_dev / rmse

def rpiq_score(predictions, targets):
    """RPIQ: Ratio of Performance to Interquartile Range."""
    iqr = np.percentile(targets, 75) - np.percentile(targets, 25)
    rmse = root_mean_squared_error(targets, predictions)
    return iqr / rmse

# Create sklearn scorers
rpd = make_scorer(rpd_score, greater_is_better=True)
rpiq = make_scorer(rpiq_score, greater_is_better=True)
mlflow_rpd_score = make_metric(eval_fn=rpd_score, greater_is_better=True, name="rpd_score")
mlflow_rpiq_score = make_metric(eval_fn=rpiq_score, greater_is_better=True, name="rpiq_score")
