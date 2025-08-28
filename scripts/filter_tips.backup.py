# -*- coding: utf-8 -*-
"""
Filtro de tips robusto.
Uso:
  python scripts/filter_tips.py INPUT.csv OUTPUT.csv --min-prob 0.60
Aceita colunas:
  - pred_prob  -> prob. de P1 (gera p2 = 1 - pred_prob)
  - p1_prob / p2_prob
  - pick       -> se não existir, escolhe P1 se p1_prob >= p2_prob
  - player1/player2 (ou p1/p2, home/away) para criar pick_name
"""
import argparse
import sys
import pandas as pd
import numpy as np
from typing import List, Optional

def find_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--min-prob", type=float, default=0.60, help="probabilidade mínima do pick (default: 0.60)")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    n0 = len(df)

    # 1) Identificar probabilidades
    pred = find_col(df, ["pred_prob", "proba", "p_win", "win_prob"])
    p1c = find_col(df, ["p1_prob", "prob_p1", "proba_p1"])
    p2c = find_col(df, ["p2_prob", "prob_p2", "proba_p2"])

    if p1c is None and p2c is None:
        if pred is None:
            raise SystemExit("Nenhuma coluna de probabilidade encontrada (procure por pred_prob ou p1_prob/p2_prob).")
        # criar p1/p2 a partir de pred_prob
        df["p1_prob"] = pd.to_numeric(df[pred], errors="coerce").fillna(0.5).clip(0, 1)
        df["p2_prob"] = 1.0 - df["p1_prob"]
        p1c, p2c = "p1_prob", "p2_prob"
    else:
        # normalizar para float e bounds
        if p1c:
            df[p1c] = pd.to_numeric(df[p1c], errors="coerce").fillna(0.5).clip(0, 1)
        if p2c:
            df[p2c] = pd.to_numeric(df[p2c], errors="coerce").fillna(0.5).clip(0, 1)
        if p1c is None and p2c is not None:
            df["p1_prob"] = 1.0 - df[p2c]
            p1c = "p1_prob"
        if p2c is None and p1c is not None:
            df["p2_prob"] = 1.0 - df[p1c]
            p2c = "p2_prob"

    # 2) Criar pick se necessário
    pick_col = find_col(df, ["pick", "selection", "side"])
    if pick_col is None:
        df["pick"] = np.where(df[p1c] >= df[p2c], "P1", "P2")
        pick_col = "pick"

    # 3) Probabilidade do pick escolhido
    df["pick_prob"] = np.where(df[pick_col].astype(str).str.upper().eq("P1"), df[p1c], df[p2c])

    # 4) Nome do pick (opcional)
    p1n = find_col(df, ["player1", "p1", "home"])
    p2n = find_col(df, ["player2", "p2", "away"])
    if p1n and p2n:
        df["pick_name"] = np.where(df[pick_col].str.upper().eq("P1"), df[p1n], df[p2n])

    # 5) Filtrar por min-prob
    df_out = df[df["pick_prob"] >= args.min_prob].copy()
    df_out = df_out.sort_values(["pick_prob"], ascending=[False])

    # 6) Guardar
    df_out.to_csv(args.out, index=False)
    print(f"[filter] {n0} -> {len(df_out)} linhas (min_prob={args.min_prob:.2f}) -> {args.out}")

if __name__ == "__main__":
    main()
