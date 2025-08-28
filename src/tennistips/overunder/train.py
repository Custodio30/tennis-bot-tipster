import argparse, joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from .dataset import build_ou_dataset

def make_features(df: pd.DataFrame):
    # features numéricas + categóricas
    num_cols = [
        "line", "best_of",
        "elo_p1","elo_p2","elo_surface_p1","elo_surface_p2",
        "fatigue_p1","fatigue_p2",
        "hold_pct_p1","hold_pct_p2","break_pct_p1","break_pct_p2",
        "tiebreak_rate_p1","tiebreak_rate_p2"
    ]
    cat_cols = ["surface","tour"]
    y = df["y_over"].astype(int).values

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ], remainder="passthrough")

    X = df[cat_cols + num_cols]
    return X, y, pre, num_cols, cat_cols

def time_split(df: pd.DataFrame, n_splits=5):
    # ordena por data e faz CV temporal
    df = df.sort_values("date")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = list(tscv.split(df))
    return df, indices

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="data/processed/matches.csv")
    ap.add_argument("--out-model", default="models/model_ou.joblib")
    ap.add_argument("--report", default="outputs/metrics/ou_cv_report.csv")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    data = build_ou_dataset(args.history)
    X, y, pre, num_cols, cat_cols = make_features(data)
    data, splits = time_split(data, n_splits=args.splits)

    # reindex X,y ao data ordenado
    X = X.loc[data.index]
    y = y[data.index]

    model_base = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.06,
        max_iter=600,
        l2_regularization=0.0
    )

    # pipeline manual: fit preprocessor -> fit calibrated model
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("pre", pre),
        ("clf", model_base)
    ])

    # calibração por split final (ou usa CV=‘prefit’ com holdout)
    # aqui, fazemos CV temporal para métricas e depois calibramos num holdout final (último split)
    rows = []
    for i, (tr, te) in enumerate(splits, start=1):
        Xt, Xv = X.iloc[tr], X.iloc[te]
        yt, yv = y[tr], y[te]

        pipe.fit(Xt, yt)
        # calibrar no train interno (platt ou isotonic). melhor: usar um pequeno holdout do Xt.
        # por simplicidade: calibrar com isotonic usando CalibratedClassifierCV(cv=3)
        calib = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
        calib.fit(Xt, yt)

        p = calib.predict_proba(Xv)[:,1]
        brier = brier_score_loss(yv, p)
        ll = log_loss(yv, p, labels=[0,1])
        rows.append({"split": i, "brier": brier, "logloss": ll})

    rep = pd.DataFrame(rows)
    rep.to_csv(args.report, index=False)

    # treina no dataset todo e calibra (cv=5 interno)
    final_calib = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
    final_calib.fit(X, y)
    joblib.dump({
        "model": final_calib,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }, args.out_model)

if __name__ == "__main__":
    main()
