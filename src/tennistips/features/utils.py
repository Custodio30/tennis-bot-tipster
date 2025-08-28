from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
from .elo import PlayerState, new_player, get_surface_elo, update_surface_elo, expected_score, k_factor

def build_features(matches: pd.DataFrame, start_elo: float, k_base: float, surface_k_boost: float, form_window: int, h2h_decay: float) -> pd.DataFrame:
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    players: Dict[str, PlayerState] = {}
    feats = []

    for _, row in df.iterrows():
        p1, p2 = str(row["player1"]), str(row["player2"])
        surf = str(row["surface"]) if pd.notna(row["surface"]) else "Unknown"
        lvl  = str(row["level"]) if pd.notna(row["level"]) else "ATP"
        date = row["date"]
        winner = str(row["winner"])

        if p1 not in players:
            players[p1] = new_player(start_elo)
        if p2 not in players:
            players[p2] = new_player(start_elo)
        s1, s2 = players[p1], players[p2]

        # Pre-game features
        elo1, elo2 = s1.elo_all, s2.elo_all
        elo1_s, elo2_s = get_surface_elo(s1, surf, start_elo), get_surface_elo(s2, surf, start_elo)
        form1 = np.mean(s1.recent_results[-form_window:]) if s1.recent_results else 0.5
        form2 = np.mean(s2.recent_results[-form_window:]) if s2.recent_results else 0.5
        h2h12 = h2h_score(s1.h2h, p2)
        h2h21 = h2h_score(s2.h2h, p1)

        feats.append({
            "date": date,
            "level": lvl,
            "surface": surf,
            "p1": p1,
            "p2": p2,
            "y": 1 if winner == p1 else 0,
            "elo_diff": elo1 - elo2,
            "elo_surf_diff": elo1_s - elo2_s,
            "form_diff": form1 - form2,
            "h2h_diff": h2h12 - h2h21,
        })

        # Update Elo after result
        s1_score = 1.0 if winner == p1 else 0.0
        s2_score = 1.0 - s1_score

        exp1_all = expected_score(elo1, elo2)
        exp1_s = expected_score(elo1_s, elo2_s)

        k1_all = k_factor(k_base, lvl, s1.matches_played, False, surface_k_boost)
        k2_all = k_factor(k_base, lvl, s2.matches_played, False, surface_k_boost)
        k1_s   = k_factor(k_base, lvl, s1.matches_played, True, surface_k_boost)
        k2_s   = k_factor(k_base, lvl, s2.matches_played, True, surface_k_boost)

        s1.elo_all = elo1 + k1_all * (s1_score - exp1_all)
        s2.elo_all = elo2 + k2_all * (s2_score - (1.0 - exp1_all))
        update_surface_elo(s1, surf, elo1_s + k1_s * (s1_score - exp1_s))
        update_surface_elo(s2, surf, elo2_s + k2_s * (s2_score - (1.0 - exp1_s)))

        s1.matches_played += 1
        s2.matches_played += 1
        s1.recent_results.append(int(s1_score))
        s2.recent_results.append(int(s2_score))
        update_h2h(s1.h2h, p2, int(s1_score), h2h_decay)
        update_h2h(s2.h2h, p1, int(s2_score), h2h_decay)

    return pd.DataFrame(feats)

def prepare_state_from_history(matches: pd.DataFrame, start_elo: float, k_base: float, surface_k_boost: float, h2h_decay: float, form_window: int):
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date")
    players: Dict[str, PlayerState] = {}

    for _, row in matches.iterrows():
        p1, p2 = str(row["player1"]), str(row["player2"])
        surf = str(row["surface"]) if pd.notna(row["surface"]) else "Unknown"
        lvl  = str(row["level"]) if pd.notna(row["level"]) else "ATP"
        winner = str(row["winner"])

        if p1 not in players:
            players[p1] = new_player(start_elo)
        if p2 not in players:
            players[p2] = new_player(start_elo)
        s1, s2 = players[p1], players[p2]

        elo1, elo2 = s1.elo_all, s2.elo_all
        elo1_s, elo2_s = get_surface_elo(s1, surf, start_elo), get_surface_elo(s2, surf, start_elo)
        s1_score = 1.0 if winner == p1 else 0.0
        s2_score = 1.0 - s1_score

        exp1_all = expected_score(elo1, elo2)
        exp1_s = expected_score(elo1_s, elo2_s)

        k1_all = k_factor(k_base, lvl, s1.matches_played, False, surface_k_boost)
        k2_all = k_factor(k_base, lvl, s2.matches_played, False, surface_k_boost)
        k1_s   = k_factor(k_base, lvl, s1.matches_played, True, surface_k_boost)
        k2_s   = k_factor(k_base, lvl, s2.matches_played, True, surface_k_boost)

        s1.elo_all = elo1 + k1_all * (s1_score - exp1_all)
        s2.elo_all = elo2 + k2_all * (s2_score - (1.0 - exp1_all))
        update_surface_elo(s1, surf, elo1_s + k1_s * (s1_score - exp1_s))
        update_surface_elo(s2, surf, elo2_s + k2_s * (s2_score - (1.0 - exp1_s)))

        s1.matches_played += 1
        s2.matches_played += 1
        s1.recent_results.append(int(s1_score))
        s2.recent_results.append(int(s2_score))
        update_h2h(s1.h2h, p2, int(s1_score), h2h_decay)
        update_h2h(s2.h2h, p1, int(s2_score), h2h_decay)

    return players

def match_features_from_state(players: Dict[str, PlayerState], p1: str, p2: str, surface: str, start_elo: float, form_window: int) -> np.ndarray:
    if p1 not in players:
        players[p1] = new_player(start_elo)
    if p2 not in players:
        players[p2] = new_player(start_elo)
    s1, s2 = players[p1], players[p2]
    elo_diff = s1.elo_all - s2.elo_all
    elo_surf_diff = get_surface_elo(s1, surface, start_elo) - get_surface_elo(s2, surface, start_elo)
    form_diff = (np.mean(s1.recent_results[-form_window:]) if s1.recent_results else 0.5) - \
                (np.mean(s2.recent_results[-form_window:]) if s2.recent_results else 0.5)
    h2h_diff = h2h_score(s1.h2h, p2) - h2h_score(s2.h2h, p1)
    return np.array([elo_diff, elo_surf_diff, form_diff, h2h_diff]).reshape(1, -1)

def update_h2h(h2h_map, opp: str, won: int, decay: float):
    w, l, wt = h2h_map.get(opp, (0, 0, 0.0))
    if won:
        w += 1
    else:
        l += 1
    wt = wt * decay + 1.0
    h2h_map[opp] = (w, l, wt)

def h2h_score(h2h_map, opp: str) -> float:
    w, l, wt = h2h_map.get(opp, (0, 0, 0.0))
    if w + l == 0:
        return 0.0
    return (w - l) / max(1.0, wt)
