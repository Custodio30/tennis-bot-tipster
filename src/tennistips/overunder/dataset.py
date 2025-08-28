import pandas as pd
import numpy as np
import re
from typing import List, Optional, Tuple

COMMON_LINES_BO3 = [20.5, 21.5, 22.5, 23.5, 24.5]
COMMON_LINES_BO5 = [35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5]

def infer_best_of(row: pd.Series) -> int:
    comp = f"{str(row.get('category','')).lower()} {str(row.get('tournament','')).lower()} {str(row.get('event','')).lower()}"
    if any(k in comp for k in ["australian open", "roland garros", "french open", "wimbledon", "us open"]):
        return 5
    try:
        return int(row.get("best_of", 3) or 3)
    except Exception:
        return 3

# -------- helpers p/ total de jogos --------

_SET_SCORE_RE = re.compile(r"(?P<a>\d+)\s*-\s*(?P<b>\d+)")

def _strip_tb(s: str) -> str:
    # remove tiebreaks: 7-6(5) -> 7-6
    return re.sub(r"\(\s*\d+\s*\)", "", s)

def total_games_from_score_text(score: str) -> Optional[int]:
    if not isinstance(score, str) or not score.strip():
        return None
    s = _strip_tb(score)
    total = 0
    for m in _SET_SCORE_RE.finditer(s):
        try:
            a = int(m.group("a")); b = int(m.group("b"))
            total += a + b
        except Exception:
            continue
    return total if total > 0 else None

def find_set_pairs_from_columns(columns: List[str]) -> List[Tuple[str, str]]:
    """
    Tenta descobrir pares (p1, p2) por set com base nos nomes das colunas.
    Exemplos suportados (case-insensitive):
      set1_p1 / set1_p2
      s1_p1 / s1_p2
      set1_home / set1_away
      set1_player1 / set1_player2
      set_1_p1 / set_1_p2
    """
    cols = [c.lower() for c in columns]
    pairs: List[Tuple[str, str]] = []
    # Construímos candidatos para até 5 sets
    patterns = [
        r"(?:set|s)[ _\-]?{i}[ _\-]?(p1|home|j1|player1|a|1)$",
        r"(?:set|s)[ _\-]?{i}[ _\-]?(p2|away|j2|player2|b|2)$",
    ]
    for i in range(1, 6):
        p1_regex = re.compile(patterns[0].format(i=i))
        p2_regex = re.compile(patterns[1].format(i=i))
        p1_col = None; p2_col = None
        for c in columns:
            cl = c.lower()
            if p1_col is None and p1_regex.search(cl):
                p1_col = c
            if p2_col is None and p2_regex.search(cl):
                p2_col = c
        if p1_col and p2_col:
            pairs.append((p1_col, p2_col))
    return pairs

def ensure_total_games(df: pd.DataFrame) -> pd.Series:
    # 0) já existe total?
    for col in ["total_games", "total", "games_total", "total_g", "totalg"]:
        if col in df.columns:
            tg = pd.to_numeric(df[col], errors="coerce")
            if tg.notna().any():
                return tg

    # 1) tenta p1/p2 totais
    candidates_p1 = ["games_p1", "p1_games", "games1", "g1", "home_games", "player1_games"]
    candidates_p2 = ["games_p2", "p2_games", "games2", "g2", "away_games", "player2_games"]
    p1 = None; p2 = None
    for c in candidates_p1:
        if c in df.columns:
            p1 = pd.to_numeric(df[c], errors="coerce"); break
    for c in candidates_p2:
        if c in df.columns:
            p2 = pd.to_numeric(df[c], errors="coerce"); break
    if p1 is not None and p2 is not None:
        tg = p1.add(p2, fill_value=np.nan)
        if tg.notna().any():
            return tg

    # 2) tenta pares por set via nomes de colunas
    pairs = find_set_pairs_from_columns(list(df.columns))
    if pairs:
        sums = []
        for idx, row in df.iterrows():
            tot = 0.0
            valid = False
            for a_col, b_col in pairs:
                a = pd.to_numeric(pd.Series([row.get(a_col)]), errors="coerce").iloc[0]
                b = pd.to_numeric(pd.Series([row.get(b_col)]), errors="coerce").iloc[0]
                if pd.notna(a) and pd.notna(b):
                    tot += float(a) + float(b)
                    valid = True
            sums.append(tot if valid else np.nan)
        return pd.Series(sums, index=df.index)

    # 3) tenta score textual em várias colunas usuais (pt/en)
    for c in ["score", "final_score", "resultado", "placar", "result", "sets", "match_score"]:
        if c in df.columns:
            return df[c].apply(total_games_from_score_text)

    # 4) falhou: devolve NaN
    return pd.Series([np.nan]*len(df), index=df.index)

# -------- dataset builder --------

def build_ou_dataset(history_csv: str,
                     lines_bo3: List[float] = None,
                     lines_bo5: List[float] = None) -> pd.DataFrame:
    if lines_bo3 is None: lines_bo3 = COMMON_LINES_BO3
    if lines_bo5 is None: lines_bo5 = COMMON_LINES_BO5

    df = pd.read_csv(history_csv)
    # normalizações
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    df["best_of_inferred"] = df.apply(infer_best_of, axis=1)
    df["total_games"] = ensure_total_games(df)

    base = df[df["total_games"].notna()].copy()
    if base.empty:
        # diagnóstico útil
        cols = list(df.columns)
        raise ValueError(
            "Não foi possível obter 'total_games' do histórico.\n"
            f"Colunas disponíveis: {cols}\n"
            "Aceites: \n"
            " - Score único: score/final_score/resultado/placar/result/sets/match_score (ex: '7-6(5) 6-7(8) 7-6(3)')\n"
            " - Por set: set1_p1/set1_p2, s1_p1/s1_p2, set1_home/set1_away, etc. (até set5_*)."
        )

    rows = []
    for _, r in base.iterrows():
        lines = lines_bo5 if int(r["best_of_inferred"]) == 5 else lines_bo3
        for L in lines:
            y = 1.0 if float(r["total_games"]) > float(L) else 0.0
            rows.append({
                "date": r.get("date"),
                "player1": r.get("player1"),
                "player2": r.get("player2"),
                "surface": r.get("surface"),
                "tour": r.get("tour"),
                "best_of": int(r["best_of_inferred"]) if pd.notna(r["best_of_inferred"]) else 3,
                "line": float(L),
                "total_games": float(r["total_games"]),
                "y_over": int(y),
                # features opcionais; ok se NaN
                "elo_p1": r.get("elo_p1", np.nan),
                "elo_p2": r.get("elo_p2", np.nan),
                "elo_surface_p1": r.get("elo_surface_p1", np.nan),
                "elo_surface_p2": r.get("elo_surface_p2", np.nan),
                "fatigue_p1": r.get("fatigue_p1", np.nan),
                "fatigue_p2": r.get("fatigue_p2", np.nan),
                "hold_pct_p1": r.get("hold_pct_p1", np.nan),
                "hold_pct_p2": r.get("hold_pct_p2", np.nan),
                "break_pct_p1": r.get("break_pct_p1", np.nan),
                "break_pct_p2": r.get("break_pct_p2", np.nan),
                "tiebreak_rate_p1": r.get("tiebreak_rate_p1", np.nan),
                "tiebreak_rate_p2": r.get("tiebreak_rate_p2", np.nan),
            })
    out = pd.DataFrame(rows)
    out["line"] = out["line"].astype(float)
    out["y_over"] = out["y_over"].astype(int)
    return out
