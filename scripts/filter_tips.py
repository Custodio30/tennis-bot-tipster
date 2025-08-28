# -*- coding: utf-8 -*-
"""
Filtro de tips com fator de notícias (lesões, withdrawals, etc.)
Uso:
  python scripts/filter_tips.py INPUT.csv OUTPUT.csv --min-prob 0.60 --news data/news/news_flags.csv --penalty 0.35 --half-life 7
Colunas esperadas (flexível):
  - Probabilidades de base:
      * pred_prob  (prob. P1)  OU
      * p1_prob / p2_prob
  - Nomes dos jogadores (para casar com notícias):
      * player1 / player2  (ou p1/p2, home/away)
  - Opcional:
      * pick  (se não existir, é recalculado)
Notícias CSV: columns ~ player,status,severity,date,detail,source
"""
import argparse
import sys
import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

# ---------- helpers ----------
def find_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None

def today_utc() -> datetime:
    return datetime.now(timezone.utc)

def parse_severity(row: Dict[str, str]) -> float:
    # prioridade: severity numérica
    sev = row.get("severity", "")
    try:
        v = float(sev)
        if 0 <= v <= 1:
            return v
    except Exception:
        pass
    # mapear por status
    status = (row.get("status", "") or "").strip().lower()
    mapping = {
        "withdrawal": 1.00, "withdrew": 1.00, "retired": 0.95, "retirement": 0.95,
        "injury": 0.85, "lesion": 0.85, "lesão": 0.85,
        "illness": 0.75, "flu": 0.70, "covid": 0.85,
        "fatigue": 0.55, "jetlag": 0.45, "travel": 0.40,
    }
    for k, v in mapping.items():
        if k in status:
            return v
    return 0.50  # default neutro-moderado

def half_life_decay(days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (days / half_life_days)

def load_news(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalizar colunas
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want in ["player","status","severity","date","detail","source"]:
        if want not in cols:
            for c in df.columns:
                if c.lower().startswith(want):
                    rename[c] = want
                    break
        else:
            rename[cols[want]] = want
    df = df.rename(columns=rename)
    for c in ["player","status","detail","source"]:
        if c not in df.columns: df[c] = ""
    if "severity" not in df.columns: df["severity"] = ""
    if "date" not in df.columns: df["date"] = ""
    # normalizar player/date
    df["player_norm"] = df["player"].astype(str).str.strip().str.lower()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df

def player_risk(player_name: str, match_dt: Optional[datetime], news_df: pd.DataFrame, half_life_days: float) -> Tuple[float, str]:
    if news_df is None or news_df.empty or not player_name:
        return 0.0, ""
    pnorm = str(player_name).strip().lower()
    rel = news_df[news_df["player_norm"] == pnorm]
    if rel.empty:
        return 0.0, ""
    base = match_dt or today_utc()
    best_risk = 0.0
    best_tag = ""
    for _, r in rel.iterrows():
        dt = r.get("date_dt")
        days = 0.0 if (pd.isna(dt) or dt is None) else max(0.0, (base - dt.to_pydatetime()).total_seconds() / 86400.0)
        sev = parse_severity({"status": str(r.get("status","")), "severity": str(r.get("severity",""))})
        risk = float(sev) * half_life_decay(days, half_life_days)
        if risk > best_risk:
            best_risk = risk
            best_tag = str(r.get("status","") or "")
    return float(best_risk), best_tag

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--min-prob", type=float, default=0.60, help="prob mínima do pick (default 0.60)")
    ap.add_argument("--news", type=str, default=None, help="CSV de notícias (player,status,severity,date,detail,source)")
    ap.add_argument("--penalty", type=float, default=0.35, help="penalização máxima aplicada (0–1), default 0.35")
    ap.add_argument("--half-life", type=float, default=7.0, help="meia-vida em dias (default 7)")
    args = ap.parse_args()

    # carregar tips
    df = pd.read_csv(args.src)
    df = df.reset_index(drop=True)
    n0 = len(df)

    # Encontrar probas base
    pred = find_col(df, ["pred_prob", "proba", "p_win", "win_prob"])
    p1c = find_col(df, ["p1_prob", "prob_p1", "proba_p1"])
    p2c = find_col(df, ["p2_prob", "prob_p2", "proba_p2"])
    if p1c is None and p2c is None:
        if pred is None:
            raise SystemExit("Nenhuma probabilidade encontrada (espera 'pred_prob' ou 'p1_prob/p2_prob').")
        df["p1_prob"] = pd.to_numeric(df[pred], errors="coerce").fillna(0.5).clip(0, 1)
        df["p2_prob"] = 1.0 - df["p1_prob"]
        p1c, p2c = "p1_prob", "p2_prob"
    else:
        if p1c:
            df[p1c] = pd.to_numeric(df[p1c], errors="coerce").fillna(0.5).clip(0, 1)
        if p2c:
            df[p2c] = pd.to_numeric(df[p2c], errors="coerce").fillna(0.5).clip(0, 1)
        if p1c is None and p2c is not None:
            df["p1_prob"] = 1.0 - df[p2c]; p1c = "p1_prob"
        if p2c is None and p1c is not None:
            df["p2_prob"] = 1.0 - df[p1c]; p2c = "p2_prob"

    # nomes jogadores
    p1n = find_col(df, ["player1", "p1", "home"])
    p2n = find_col(df, ["player2", "p2", "away"])

    # data do jogo (opcional)
    dcol = find_col(df, ["start_time","match_date","date","start","kickoff"])
    match_dates = pd.to_datetime(df[dcol], errors="coerce", utc=True) if dcol else pd.Series([pd.NaT]*len(df))

    # carregar noticias (gracioso se faltar)
    news_df = None
    if args.news:
        try:
            news_df = load_news(args.news)
        except Exception as e:
            print(f"[filter] aviso: falha a ler notícias: {e}", file=sys.stderr)

    # riscos por jogador
    r1_list, r1_tag, r2_list, r2_tag = [], [], [], []
    if news_df is not None and not news_df.empty and p1n and p2n:
        for i in range(len(df)):
            dt = None
            if dcol and not pd.isna(match_dates.iloc[i]):
                dt = match_dates.iloc[i].to_pydatetime()
            nm1 = str(df.iloc[i][p1n]) if p1n else ""
            nm2 = str(df.iloc[i][p2n]) if p2n else ""
            r1, t1 = player_risk(nm1, dt, news_df, args.half_life)
            r2, t2 = player_risk(nm2, dt, news_df, args.half_life)
            r1_list.append(r1); r1_tag.append(t1)
            r2_list.append(r2); r2_tag.append(t2)
    else:
        r1_list = [0.0]*len(df); r2_list = [0.0]*len(df)
        r1_tag  = [""  ]*len(df); r2_tag  = [""  ]*len(df)

    df["news_risk_p1"] = r1_list
    df["news_risk_p2"] = r2_list
    df["news_tag_p1"]  = r1_tag
    df["news_tag_p2"]  = r2_tag

    # aplicar penalização simétrica e renormalizar
    pen = float(args.penalty)
    p1a = df[p1c].to_numpy(dtype=float) * (1.0 - pen * df["news_risk_p1"].to_numpy(dtype=float))
    p2a = df[p2c].to_numpy(dtype=float) * (1.0 - pen * df["news_risk_p2"].to_numpy(dtype=float))
    s = p1a + p2a
    s[s <= 0] = 1.0  # evitar divisão por zero
    p1_adj = p1a / s
    p2_adj = p2a / s

    df["p1_prob_adj"] = p1_adj
    df["p2_prob_adj"] = p2_adj

    # pick e prob ajustados
    df["pick"] = np.where(df["p1_prob_adj"] >= df["p2_prob_adj"], "P1", "P2")
    df["pick_prob"] = np.where(df["pick"].eq("P1"), df["p1_prob_adj"], df["p2_prob_adj"])

    # nome do pick (se possível)
    if p1n and p2n:
        df["pick_name"] = np.where(df["pick"].eq("P1"), df[p1n], df[p2n])

    # --------- FILTRO com min_prob (corrigido) ---------
    min_prob = float(args.min_prob)
    mask = df["pick_prob"] >= min_prob
    df_out = df[mask].copy()
    df_out = df_out.sort_values(["pick_prob"], ascending=[False])

    # guardar
    df_out.to_csv(args.out, index=False)
    print(f"[filter] {n0} -> {len(df_out)} linhas | min_prob={min_prob:.2f} | news={'ON' if (news_df is not None and not news_df.empty) else 'OFF'} -> {args.out}")

if __name__ == "__main__":
    main()
