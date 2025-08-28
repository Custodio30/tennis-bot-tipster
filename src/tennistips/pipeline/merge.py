import os
import pandas as pd
import numpy as np
import yaml
from rapidfuzz import fuzz

def _norm_name(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = x.replace(".", "").replace("-", " ")
    x = " ".join(x.split())
    return x

def load_sackmann_raw(raw_dir: str, min_year: int = 2000) -> pd.DataFrame:
    files = [f for f in os.listdir(raw_dir) if f.startswith("atp_matches_") and f.endswith(".csv")]
    dfs = []
    for f in files:
        try:
            year = int(f.split("_")[-1].split(".")[0])
        except Exception:
            continue
        if year < min_year:
            continue
        df = pd.read_csv(os.path.join(raw_dir, f))
        keep = {
            "tourney_date": "date",
            "surface": "surface",
            "winner_name": "winner",
            "loser_name": "loser",
            "round": "round",
            "tourney_name": "tournament",
            "tourney_level": "level",
        }
        # apenas prossegue se todas as colunas existem
        if not set(keep.keys()).issubset(df.columns):
            continue
        df = df[list(keep.keys())].rename(columns=keep)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["date", "surface", "winner", "loser", "round", "tournament", "level"])
    return pd.concat(dfs, ignore_index=True)

def load_tennisdata_raw(raw_dir: str) -> pd.DataFrame:
    files = [f for f in os.listdir(raw_dir) if f.startswith("tennisdata_atp_") and f.endswith(".csv")]
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(raw_dir, f))
        # campos esperados no Tennis-Data
        rename_map = {
            "Date": "date",
            "Surface": "surface",
            "Tournament": "tournament",
            "Winner": "winner",
            "Loser": "loser",
            "Round": "round",
        }
        if not set(rename_map.keys()).issubset(df.columns):
            # ficheiro fora do padrão: ignora e continua
            continue
        df = df.rename(columns=rename_map)
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

        # odds: prioriza Bet365, senão Pinnacle/PS
        if "B365W" in df.columns and "B365L" in df.columns:
            df["odds_w"] = df["B365W"]; df["odds_l"] = df["B365L"]
        elif "PSW" in df.columns and "PSL" in df.columns:
            df["odds_w"] = df["PSW"]; df["odds_l"] = df["PSL"]
        else:
            df["odds_w"] = np.nan; df["odds_l"] = np.nan

        dfs.append(df[["date", "surface", "tournament", "winner", "loser", "round", "odds_w", "odds_l"]])

    if not dfs:
        return pd.DataFrame(columns=["date", "surface", "tournament", "winner", "loser", "round", "odds_w", "odds_l"])
    return pd.concat(dfs, ignore_index=True)

def fuzzy_merge_results_and_odds(results: pd.DataFrame, odds: pd.DataFrame, threshold: int = 92) -> pd.DataFrame:
    # Sem resultados → nada a fazer
    if results.empty:
        return pd.DataFrame(columns=["date", "surface", "tournament", "player1", "player2", "winner", "odds_p1", "odds_p2"])
    # Sem odds → devolve vazio para forçar fallback no build_merged_dataset
    if odds.empty:
        return pd.DataFrame(columns=["date", "surface", "tournament", "player1", "player2", "winner", "odds_p1", "odds_p2"])

    res = results.copy()
    odd = odds.copy()

    # chaves de data (ambas já são datetime pelas loaders)
    res["key_date"] = res["date"].dt.date.astype("string")
    odd["key_date"] = odd["date"].dt.date.astype("string")

    # normalização de nomes
    res["p1"] = res["winner"]
    res["p2"] = res["loser"]
    res["p1_norm"] = res["p1"].map(_norm_name)
    res["p2_norm"] = res["p2"].map(_norm_name)
    odd["w_norm"] = odd["winner"].map(_norm_name)
    odd["l_norm"] = odd["loser"].map(_norm_name)

    merged_rows = []
    odd_by_date = {d: g.reset_index(drop=True) for d, g in odd.groupby("key_date")}

    for _, row in res.iterrows():
        d = row["key_date"]
        candidates = odd_by_date.get(d)
        if candidates is None or candidates.empty:
            continue

        best_idx = -1
        best_score = -1
        for j, cand in candidates.iterrows():
            s1 = fuzz.ratio(row["p1_norm"], cand["w_norm"])
            s2 = fuzz.ratio(row["p2_norm"], cand["l_norm"])
            s_rev1 = fuzz.ratio(row["p1_norm"], cand["l_norm"])
            s_rev2 = fuzz.ratio(row["p2_norm"], cand["w_norm"])
            score = max(min(s1, s2), min(s_rev1, s_rev2))
            if score > best_score:
                best_score = score; best_idx = j

        if best_score >= threshold:
            cand = candidates.loc[best_idx]
            # mapear odds mantendo player1 como o winner em 'res'
            if fuzz.ratio(row["p1_norm"], cand["w_norm"]) >= fuzz.ratio(row["p1_norm"], cand["l_norm"]):
                o1, o2 = cand.get("odds_w", np.nan), cand.get("odds_l", np.nan)
            else:
                o1, o2 = cand.get("odds_l", np.nan), cand.get("odds_w", np.nan)

            merged_rows.append({
                "date": row["date"],
                "tournament": row["tournament"],
                "surface": row["surface"],
                "player1": row["p1"],
                "player2": row["p2"],
                "winner": row["p1"],
                "odds_p1": o1,
                "odds_p2": o2
            })

    return pd.DataFrame(merged_rows)

def build_merged_dataset(cfg_path: str) -> str:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    raw_dir = cfg["paths"]["raw_dir"]
    out_csv = cfg["paths"]["merged_csv"]

    res = load_sackmann_raw(raw_dir)
    odds = load_tennisdata_raw(raw_dir)

    if res.empty:
        raise RuntimeError("Nenhum resultado carregado do Sackmann. Já correste o fetch?")

    if odds.empty:
        merged = res.copy()
        merged["player1"] = res["winner"]
        merged["player2"] = res["loser"]
        merged["odds_p1"] = np.nan
        merged["odds_p2"] = np.nan
    else:
        merged = fuzzy_merge_results_and_odds(res, odds)
        if merged.empty:
         merged = res.copy()
         merged["player1"] = res["winner"]
         merged["player2"] = res["loser"]
         merged["odds_p1"] = np.nan
         merged["odds_p2"] = np.nan


    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return out_csv
