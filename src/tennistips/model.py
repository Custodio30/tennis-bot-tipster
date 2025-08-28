from typing import Tuple, Dict, Any, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
import joblib


# ---------- Compat wrapper (sempre expõe predict_proba) ----------

class ProbModelWrapper:
    """
    Fornece uma interface unificada .predict_proba(X)->(n,2)
    kind: "logreg" | "hgb" | "ensemble"
    obj:
      - logreg: CalibratedClassifierCV
      - hgb: (HistGradientBoostingClassifier, IsotonicRegression)
      - ensemble: {"lr": ProbModelWrapper, "hgb": ProbModelWrapper, "w": float}
    """
    def __init__(self, kind: str, obj: Any):
        self.kind = kind
        self.obj = obj

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "logreg":
            p1 = self.obj.predict_proba(X)[:, 1]
        elif self.kind == "hgb":
            base, iso = self.obj
            p_raw = base.predict_proba(X)[:, 1]
            p1 = iso.predict(p_raw)
        elif self.kind == "ensemble":
            w = float(self.obj["w"])
            p_lr = self.obj["lr"].predict_proba(X)[:, 1]
            p_hg = self.obj["hgb"].predict_proba(X)[:, 1]
            p1 = w * p_lr + (1.0 - w) * p_hg
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])


# ---------- Legacy function (holdout simples) ----------

def train_logreg_calibrated(
    X: np.ndarray, y: np.ndarray, test_size: float, shuffle: bool, max_iter: int, calibration: str, seed: int
) -> Tuple[ProbModelWrapper, Dict[str, float]]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=seed
    )
    base = LogisticRegression(max_iter=max_iter, random_state=seed)
    model = CalibratedClassifierCV(base, method=calibration, cv=3)
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]
    metrics = {
        "log_loss": float(log_loss(y_val, p_val)),
        "auc": float(roc_auc_score(y_val, p_val)),
        "samples_train": int(len(X_train)),
        "samples_val": int(len(X_val)),
    }
    return ProbModelWrapper("logreg", model), metrics


# ---------- Melhor: TimeSeriesSplit + calibração ----------

def train_logreg_ts_cv(
    X: np.ndarray, y: np.ndarray, max_iter: int = 2000, calibration: str = "isotonic",
    n_splits: int = 5, seed: int = 42
) -> Dict[str, Any]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, lls = [], []

    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        base = LogisticRegression(max_iter=max_iter, random_state=seed, C=1.5)
        model = CalibratedClassifierCV(base, method=calibration, cv=3)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, p))
        lls.append(log_loss(y_va, p))

    # retrain full
    base = LogisticRegression(max_iter=max_iter, random_state=seed, C=1.5)
    final = CalibratedClassifierCV(base, method=calibration, cv=3).fit(X, y)
    wrapper = ProbModelWrapper("logreg", final)

    return {"model": wrapper, "cv_auc": float(np.mean(aucs)), "cv_logloss": float(np.mean(lls))}


def train_hgb_ts_cv(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42,
    max_depth: int = 6, learning_rate: float = 0.06, max_iter: int = 400
) -> Dict[str, Any]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, lls = [], []

    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        base = HistGradientBoostingClassifier(
            max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
            l2_regularization=0.0, random_state=seed
        )
        base.fit(X_tr, y_tr)
        p_raw = base.predict_proba(X_va)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw, y_va)
        p = iso.predict(p_raw)
        aucs.append(roc_auc_score(y_va, p))
        lls.append(log_loss(y_va, p))

    # retrain full + isotonic on full (alternativa: calibrar numa janela final)
    base = HistGradientBoostingClassifier(
        max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
        l2_regularization=0.0, random_state=seed
    ).fit(X, y)
    p_full = base.predict_proba(X)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_full, y)
    wrapper = ProbModelWrapper("hgb", (base, iso))

    return {"model": wrapper, "cv_auc": float(np.mean(aucs)), "cv_logloss": float(np.mean(lls))}


# ---------- Utilidades de ensemble (média das probs) ----------

def ensemble_avg(p1: np.ndarray, p2: np.ndarray, w1: float = 0.5) -> np.ndarray:
    return w1 * p1 + (1.0 - w1) * p2


# ---------- Persistência ----------

def save_model(model: Union[ProbModelWrapper, Tuple[str, dict]], path: str):
    """
    Aceita tanto um ProbModelWrapper como um tuple ("ensemble", {...}).
    Se receber o tuple, converte para ProbModelWrapper de ensemble.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if isinstance(model, ProbModelWrapper):
        joblib.dump(model, path)
        return

    # tuple estilo ("ensemble", {"lr": wrapper, "hgb": wrapper, "w": 0.5})
    if isinstance(model, tuple) and model[0] == "ensemble":
        lr = model[1]["lr"]
        hgb = model[1]["hgb"]
        w = float(model[1]["w"])
        # Garante que são wrappers (se vierem crus, embrulha)
        if not isinstance(lr, ProbModelWrapper):
            lr = ProbModelWrapper("logreg", lr)
        if not isinstance(hgb, ProbModelWrapper):
            hgb = ProbModelWrapper("hgb", hgb)
        wrapper = ProbModelWrapper("ensemble", {"lr": lr, "hgb": hgb, "w": w})
        joblib.dump(wrapper, path)
        return

    # fallback: embrulha tudo como logreg
    wrapper = ProbModelWrapper("logreg", model)
    joblib.dump(wrapper, path)


def load_model(path: str) -> ProbModelWrapper:
    return joblib.load(path)
