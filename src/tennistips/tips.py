
import pandas as pd
import numpy as np
from .features import prepare_state_from_history, match_features_from_state

def kelly_fraction(p: float, odds: float) -> float:
    b = odds - 1.0
    edge = p*odds - (1 - p)
    if b <= 0:
        return 0.0
    f = edge / b
    return max(0.0, f)

def generate_tips(history_df: pd.DataFrame, fixtures_df: pd.DataFrame, model, cfg) -> pd.DataFrame:
    players = prepare_state_from_history(history_df, cfg.elo.start, cfg.elo.k_base, cfg.elo.surface_k_boost, cfg.features.h2h_decay, cfg.features.form_window)

    rows = []
    for _, r in fixtures_df.iterrows():
        p1, p2 = str(r["player1"]), str(r["player2"])
        surface = str(r["surface"]) if pd.notna(r["surface"]) else "Unknown"
        o1, o2 = float(r["odds_p1"]), float(r["odds_p2"])

        X = match_features_from_state(players, p1, p2, surface, cfg.elo.start, cfg.features.form_window)
        p1_win = float(model.predict_proba(X)[:,1])
        p2_win = 1 - p1_win

        ev1 = p1_win * o1 - (1 - p1_win)
        ev2 = p2_win * o2 - (1 - p2_win)

        k1 = min(cfg.selection.kelly_cap, cfg.selection.kelly_fraction * kelly_fraction(p1_win, o1))
        k2 = min(cfg.selection.kelly_cap, cfg.selection.kelly_fraction * kelly_fraction(p2_win, o2))

        pick = "P1" if ev1 > ev2 else "P2"
        best_ev = max(ev1, ev2)
        stake = k1 if pick == "P1" else k2

        if best_ev >= cfg.selection.ev_threshold:
            rows.append({
                "player1": p1, "player2": p2, "surface": surface,
                "odds_p1": o1, "odds_p2": o2,
                "p1_prob": round(p1_win, 4), "p2_prob": round(p2_win, 4),
                "ev_p1": round(ev1, 4), "ev_p2": round(ev2, 4),
                "pick": pick, "best_ev": round(best_ev, 4),
                "stake_suggest": round(stake, 4)
            })

    tips = pd.DataFrame(rows).sort_values("best_ev", ascending=False).reset_index(drop=True)
    return tips
