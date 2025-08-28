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
      - hgb: (HistGradientBoostingClassifier, IsotonicRegression)  OU  qualquer estimador com predict_proba
      - ensemble: {"lr": ProbModelWrapper, "hgb": ProbModelWrapper, "w": float}
    """
    def __init__(self, kind: str, obj: Any):
        self.kind = kind
        self.obj = obj

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "logreg":
            p1 = self.obj.predict_proba(X)[:, 1]

        elif self.kind == "hgb":
            # Aceita tupla (base, iso) OU objeto único com predict_proba
            if isinstance(self.obj, tuple) and len(self.obj) == 2:
                base, iso = self.obj
                p_raw = base.predict_proba(X)[:, 1]
                p1 = iso.predict(p_raw)
            else:
                proba = self.obj.predict_proba(X)
                # se vier (n,2), usa a coluna 1; se vier (n,), aceita direto
                p1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else np.ravel(proba)

        elif self.kind == "ensemble":
            w = float(self.obj["w"])
            p_lr = self.obj["lr"].predict_proba(X)[:, 1]
            p_hg = self.obj["hgb"].predict_proba(X)[:, 1]
            p1 = w * p_lr + (1.0 - w) * p_hg

        else:
            raise ValueError(f"Unknown model kind: {self.kind}")

        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])

# ---------- Persistência ----------

def save_model(model: Union['ProbModelWrapper', Tuple[str, dict]], path: str):
    """
    Aceita tanto um ProbModelWrapper como um tuple ("ensemble", {...}).
    Se receber o tuple, converte para ProbModelWrapper de ensemble.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # já é wrapper? salva direto
    if isinstance(model, ProbModelWrapper):
        joblib.dump(model, path)
        return

    # tuple estilo ("ensemble", {"lr": wrapper/estimador, "hgb": wrapper/estimador, "w": 0.5})
    if isinstance(model, tuple) and model[0] == "ensemble":
        lr = model[1]["lr"]
        hgb = model[1]["hgb"]
        w = float(model[1]["w"])

        # Garante wrappers coerentes
        if not isinstance(lr, ProbModelWrapper):
            lr = ProbModelWrapper("logreg", lr)
        if not isinstance(hgb, ProbModelWrapper):
            # não sabemos se é tupla (base, iso) ou objeto, mas o wrapper lida com ambos
            hgb = ProbModelWrapper("hgb", hgb)

        wrapper = ProbModelWrapper("ensemble", {"lr": lr, "hgb": hgb, "w": w})
        joblib.dump(wrapper, path)
        return

    # fallback: embrulha como logreg (mantém retrocompat)
    wrapper = ProbModelWrapper("logreg", model)
    joblib.dump(wrapper, path)

def load_model(path: str) -> ProbModelWrapper:
    return joblib.load(path)


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
    X, y, n_splits=5, seed=42, max_iter=2000, calibration="isotonic"
):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, log_loss

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, lls = [], []
    valid_folds = 0

    for tr_idx, va_idx in tscv.split(X):
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Skip folds sem as duas classes (evita o erro do solver/calibrator)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        base = LogisticRegression(max_iter=max_iter, random_state=seed, C=1.5)
        model = CalibratedClassifierCV(base, method=calibration, cv=3)
        model.fit(X[tr_idx], y_tr)

        p = model.predict_proba(X[va_idx])[:, 1]
        aucs.append(roc_auc_score(y_va, p))
        lls.append(log_loss(y_va, p))
        valid_folds += 1

    # Treino final no conjunto completo (com salvaguarda caso falte classe)
    if len(np.unique(y)) < 2:
        # Último recurso: sem calibração (pelo menos não rebenta)
        final_model = LogisticRegression(max_iter=max_iter, random_state=seed, C=1.5)
        final_model.fit(X, y)
    else:
        final_model = CalibratedClassifierCV(
            LogisticRegression(max_iter=max_iter, random_state=seed, C=1.5),
            method=calibration, cv=3
        )
        final_model.fit(X, y)

    out = {
        "model": final_model,
        "cv_auc": float(np.mean(aucs)) if valid_folds > 0 else float("nan"),
        "cv_logloss": float(np.mean(lls)) if valid_folds > 0 else float("nan"),
        "n_valid_folds": valid_folds,
    }
    return out


def train_hgb_ts_cv(
    X, y,
    n_splits=5,
    seed=42,
    max_depth=6,
    learning_rate=0.06,
    max_iter=2000
):
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.base import BaseEstimator, ClassifierMixin

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, lls = [], []
    valid_folds = 0

    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # precisa de 2 classes no train e no val
        if np.unique(y_tr).size < 2 or np.unique(y_va).size < 2:
            continue

        base = HistGradientBoostingClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=seed
        )
        base.fit(X_tr, y_tr)

        # calibrar na própria validação (isotonic)
        p_raw = base.predict_proba(X_va)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw, y_va)
        p = iso.predict(p_raw)

        # métricas só fazem sentido com 2 classes em y_va (já garantido acima)
        aucs.append(roc_auc_score(y_va, p))
        # log_loss binário espera probs das duas classes (N,2)
        lls.append(log_loss(y_va, np.c_[1 - p, p]))
        valid_folds += 1

    # treina base final em tudo
    full_base = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=seed
    ).fit(X, y)

    # tentar calibrar em uma janela final que tenha 0 e 1
    def _calibrate_on_tail(base, X_all, y_all):
        win = min(len(y_all), 50000)  # começa com janela grande
        while win >= 1000:
            y_tail = y_all[-win:]
            if np.unique(y_tail).size == 2:
                p_tail = base.predict_proba(X_all[-win:])[:, 1]
                iso_full = IsotonicRegression(out_of_bounds="clip").fit(p_tail, y_tail)

                class CalibratedHGB(BaseEstimator, ClassifierMixin):
                    def __init__(self, base, iso):
                        self.base = base
                        self.iso = iso
                    def predict_proba(self, X_):
                        p_raw_ = self.base.predict_proba(X_)[:, 1]
                        p_ = self.iso.predict(p_raw_)
                        return np.c_[1 - p_, p_]

                return CalibratedHGB(base, iso_full)
            win //= 2
        return None  # não foi possível calibrar

    calibrated = _calibrate_on_tail(full_base, X, y)
    model = calibrated if calibrated is not None else full_base

    return {
        "model": model,
        "cv_auc": float(np.nanmean(aucs)) if len(aucs) else float("nan"),
        "cv_logloss": float(np.nanmean(lls)) if len(lls) else float("nan"),
        "n_valid_folds": valid_folds
    }

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
