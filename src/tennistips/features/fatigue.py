# src/tennistips/features/fatigue.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FatigueParams:
    # pesos lineares simples; ajusta à vontade
    w_m7: float = 0.015      # penalização por jogo nos últimos 7 dias
    w_m14: float = 0.010     # penalização extra por jogo nos últimos 14 dias
    w_m30: float = 0.005     # penalização extra por jogo nos últimos 30 dias
    b2b: float = 0.030       # back-to-back (<=1 dia)
    w_48h: float = 0.015     # descanso curto (<=2 dias)
    min_p: float = 0.05      # limites de prob após ajuste
    max_p: float = 0.95

def _count_matches(last_dates: pd.Series, ref_date: pd.Timestamp) -> Dict[str, int]:
    m7  = (last_dates >= (ref_date - pd.Timedelta(days=7))).sum()
    m14 = (last_dates >= (ref_date - pd.Timedelta(days=14))).sum()
    m30 = (last_dates >= (ref_date - pd.Timedelta(days=30))).sum()
    return {"m7": int(m7), "m14": int(m14), "m30": int(m30)}

def _player_slice(history_df: pd.DataFrame, player: str) -> pd.DataFrame:
    return history_df[(history_df["player1"] == player) | (history_df["player2"] == player)]

def _player_fatigue(history_df: pd.DataFrame, player: str, ref_date: pd.Timestamp) -> Dict[str, Any]:
    h = _player_slice(history_df, player)
    if h.empty:
        return {
            "matches_7d": 0, "matches_14d": 0, "matches_30d": 0,
            "days_since_last": 9999, "b2b": 0, "rest_48h": 0
        }
    dates = pd.to_datetime(h["date"])
    counts = _count_matches(dates, ref_date)
    last_dt = dates.max()
    days_since = int((ref_date.normalize() - last_dt.normalize()).days)
    return {
        "matches_7d": counts["m7"],
        "matches_14d": counts["m14"],
        "matches_30d": counts["m30"],
        "days_since_last": days_since,
        "b2b": 1 if days_since <= 1 else 0,
        "rest_48h": 1 if days_since <= 2 else 0,
    }

def add_fatigue_features(fixtures_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas p1_/p2_ com métricas de fadiga calculadas a partir do history.
    Requer colunas: fixtures -> [date, player1, player2], history -> [date, player1, player2].
    """
    fx = fixtures_df.copy()
    fx["date"] = pd.to_datetime(fx["date"])
    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"])

    p1_feats = []
    p2_feats = []
    for _, r in fx.iterrows():
        ref_date = pd.to_datetime(r["date"])
        p1 = str(r["player1"])
        p2 = str(r["player2"])
        p1_feats.append(_player_fatigue(history, p1, ref_date))
        p2_feats.append(_player_fatigue(history, p2, ref_date))

    p1_df = pd.DataFrame(p1_feats).add_prefix("p1_")
    p2_df = pd.DataFrame(p2_feats).add_prefix("p2_")
    out = pd.concat([fx.reset_index(drop=True), p1_df, p2_df], axis=1)
    return out

def _penalty(f: Dict[str, Any], params: FatigueParams) -> float:
    pen = 0.0
    pen += params.w_m7  * f.get("matches_7d", 0)
    pen += params.w_m14 * max(0, f.get("matches_14d", 0) - f.get("matches_7d", 0))
    pen += params.w_m30 * max(0, f.get("matches_30d", 0) - f.get("matches_14d", 0))
    if f.get("b2b", 0):    pen += params.b2b
    if f.get("rest_48h", 0) and not f.get("b2b", 0): pen += params.w_48h
    return pen

def adjust_probs_for_fixtures_row(row: pd.Series, p1_prob: float, p2_prob: float, params: FatigueParams = FatigueParams()):
    """
    Ajusta p1_prob/p2_prob penalizando o lado mais fatigado e renormaliza.
    Limita ao intervalo [min_p, max_p].
    """
    f1 = {
        "matches_7d": row.get("p1_matches_7d", 0),
        "matches_14d": row.get("p1_matches_14d", 0),
        "matches_30d": row.get("p1_matches_30d", 0),
        "b2b": row.get("p1_b2b", 0),
        "rest_48h": row.get("p1_rest_48h", 0),
    }
    f2 = {
        "matches_7d": row.get("p2_matches_7d", 0),
        "matches_14d": row.get("p2_matches_14d", 0),
        "matches_30d": row.get("p2_matches_30d", 0),
        "b2b": row.get("p2_b2b", 0),
        "rest_48h": row.get("p2_rest_48h", 0),
    }
    p = params
    pen1 = _penalty(f1, p)
    pen2 = _penalty(f2, p)

    p1 = max(p.min_p, min(p.max_p, p1_prob - pen1))
    p2 = max(p.min_p, min(p.max_p, p2_prob - pen2))

    # renormaliza para somar ≈ 1
    s = p1 + p2
    if s > 0:
        p1, p2 = p1 / s, p2 / s
    return float(p1), float(p2)
