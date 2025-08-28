
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import joblib

def train_logreg_calibrated(X: np.ndarray, y: np.ndarray, test_size: float, shuffle: bool, max_iter: int, calibration: str, seed: int):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=seed
    )
    base = LogisticRegression(max_iter=max_iter, random_state=seed)
    model = CalibratedClassifierCV(base, method=calibration, cv=3)
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:,1]
    metrics = {
        "log_loss": float(log_loss(y_val, p_val)),
        "auc": float(roc_auc_score(y_val, p_val)),
        "samples_train": int(len(X_train)),
        "samples_val": int(len(X_val)),
    }
    return model, metrics

def save_model(model, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
