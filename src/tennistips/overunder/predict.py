import argparse, joblib, pandas as pd
from .dataset import COMMON_LINES_BO3, COMMON_LINES_BO5, infer_best_of

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="data/fixtures/latest_for_tips.csv")
    ap.add_argument("--model", default="models/model_ou.joblib")
    ap.add_argument("--out", default="outputs/totals_model.csv")
    ap.add_argument("--lines", default="")  # se vazio, usa comuns por BO
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]

    fx = pd.read_csv(args.fixtures)
    fx["best_of_inferred"] = fx.apply(infer_best_of, axis=1)

    rows = []
    for _, r in fx.iterrows():
        lines = (list(map(float, args.lines.split(","))) if args.lines
                 else (COMMON_LINES_BO5 if r["best_of_inferred"]==5 else COMMON_LINES_BO3))
        for L in lines:
            row = {
                # repetir lógica de features que usaste no treino:
                "surface": r.get("surface"),
                "tour": r.get("tour"),
                "best_of": int(r["best_of_inferred"]),
                "line": float(L),
                "elo_p1": r.get("elo_p1"),
                "elo_p2": r.get("elo_p2"),
                "elo_surface_p1": r.get("elo_surface_p1"),
                "elo_surface_p2": r.get("elo_surface_p2"),
                "fatigue_p1": r.get("fatigue_p1"),
                "fatigue_p2": r.get("fatigue_p2"),
                "hold_pct_p1": r.get("hold_pct_p1"),
                "hold_pct_p2": r.get("hold_pct_p2"),
                "break_pct_p1": r.get("break_pct_p1"),
                "break_pct_p2": r.get("break_pct_p2"),
                "tiebreak_rate_p1": r.get("tiebreak_rate_p1"),
                "tiebreak_rate_p2": r.get("tiebreak_rate_p2"),
            }
            X = pd.DataFrame([row])
            prob_over = float(model.predict_proba(X)[:,1][0])
            rows.append({
                "match_id": r.get("match_id"),
                "player1": r.get("player1"),
                "player2": r.get("player2"),
                "surface": r.get("surface"),
                "best_of": int(r["best_of_inferred"]),
                "line": L,
                "prob_over": prob_over,
                "prob_under": 1.0 - prob_over,
            })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
