# -*- coding: utf-8 -*-
"""
Gerar picks de Totais (Over/Under) de jogos, com ajuste opcional por notícias e filtros de torneio.

Exemplos:
  # Só Grand Slam ATP, BO5, OVER 37.5
  python scripts/generate_overunders.py outputs/tips.csv outputs/totals.csv \
    --line 37.5 --side over --tour atp --categories gs --best-of 5 --min-prob 0.60

  # ATP Tour (1000/500/250), qualquer BO, várias linhas
  python scripts/generate_overunders.py outputs/tips.csv outputs/totals.csv \
    --lines 20.5,21.5,22.5,23.5 --tour atp --categories 1000,500,250 --min-prob 0.60
"""
import argparse, re, math
from datetime import datetime, timezone
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

# -------------- utils --------------

def find_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def today_utc() -> datetime:
    return datetime.now(timezone.utc)

def half_life_decay(days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (days / half_life_days)

def parse_severity(row: Dict[str, str]) -> float:
    sev = row.get("severity", "")
    try:
        v = float(sev)
        if 0 <= v <= 1:
            return v
    except Exception:
        pass
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
    return 0.50

def load_news(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    df["player_norm"] = df["player"].astype(str).str.strip().str.lower()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df

def player_risk(player_name: str, match_dt: Optional[datetime], news_df: pd.DataFrame, half_life_days: float) -> float:
    if news_df is None or news_df.empty or not player_name:
        return 0.0
    pnorm = str(player_name).strip().lower()
    rel = news_df[news_df["player_norm"] == pnorm]
    if rel.empty:
        return 0.0
    base = match_dt or today_utc()
    best = 0.0
    for _, r in rel.iterrows():
        dt = r.get("date_dt")
        days = 0.0 if (pd.isna(dt) or dt is None) else max(0.0, (base - dt.to_pydatetime()).total_seconds()/86400.0)
        sev = parse_severity({"status": str(r.get("status","")), "severity": str(r.get("severity",""))})
        best = max(best, sev * half_life_decay(days, half_life_days))
    return float(best)

def parse_surface(val: str) -> str:
    s = (str(val) or "").lower()
    if "clay" in s or "terra" in s: return "clay"
    if "grass" in s or "relva" in s: return "grass"
    if "carpet" in s: return "carpet"
    return "hard"

def detect_tour_from_text(t_name: str, tour_val: str) -> str:
    s = ((tour_val or "") + " " + (t_name or "")).lower()
    if "wta" in s or "women" in s or "ladies" in s:
        return "wta"
    if "atp" in s or "challenger" in s or "masters" in s or "men" in s:
        return "atp"
    return "unknown"

def detect_category(t_name: str, level_val: str) -> str:
    s = ((level_val or "") + " " + (t_name or "")).lower()
    if any(x in s for x in ["grand slam","australian open","roland garros","french open","wimbledon","us open"]):
        return "gs"
    if "masters" in s or "1000" in s or "atp 1000" in s or "m1000" in s:
        return "1000"
    if "500" in s or "atp 500" in s:
        return "500"
    if "250" in s or "atp 250" in s:
        return "250"
    if "challenger" in s:
        return "challenger"
    if "itf" in s:
        return "itf"
    return "other"

# ---------- helpers p/ inferir best-of e tour em GS ----------

def infer_best_of(cat: str, bo_num: int, round_text: str) -> int:
    """
    Se bo_num==0 e for GS, assume 5 exceto qualificações (3).
    Caso contrário, devolve 3 por omissão.
    """
    if bo_num in (3, 5):
        return bo_num
    if cat == "gs":
        if isinstance(round_text, str) and re.search(r"qual", round_text, flags=re.I):
            return 3
        return 5
    return 3 if bo_num == 0 else bo_num

def adjust_tour_if_gs(tour_tag: str, cat: str, best_of: int) -> str:
    """
    Se o tour veio 'unknown' mas é GS, usa BO5->ATP e BO3->WTA.
    """
    if tour_tag != "unknown":
        return tour_tag
    if cat == "gs":
        return "atp" if best_of >= 5 else "wta"
    return tour_tag

# -------------- main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--lines", type=str, default="20.5,21.5,22.5,23.5")
    ap.add_argument("--line", type=float, default=None, help="linha única (ex.: 37.5)")
    ap.add_argument("--side", type=str, default="both", choices=["over","under","both"], help="filtrar lado")
    ap.add_argument("--min-prob", type=float, default=0.60)
    ap.add_argument("--news", type=str, default=None)
    ap.add_argument("--half-life", type=float, default=7.0)
    ap.add_argument("--news-gamma", type=float, default=1.5, help="quanto os riscos baixam a média (jogos)")

    # parâmetros do modelo de totais
    ap.add_argument("--mu-hard", type=float, default=22.0)
    ap.add_argument("--mu-clay", type=float, default=22.6)
    ap.add_argument("--mu-grass", type=float, default=21.8)
    ap.add_argument("--mu-carpet", type=float, default=21.0)
    ap.add_argument("--sigma", type=float, default=4.5)
    ap.add_argument("--comp-gain", type=float, default=4.0, help="ganho em jogos por (competitividade-0.5)")
    ap.add_argument("--bo5-bonus", type=float, default=12.0)

    # filtros
    ap.add_argument("--tour", type=str, default="both", choices=["atp","wta","both"])
    ap.add_argument("--categories", type=str, default="", help="lista: gs,1000,500,250,challenger,itf")
    ap.add_argument("--best-of", type=int, default=None, help="3 ou 5")
    ap.add_argument("--name-like", type=str, default="", help="regex para nome do torneio")
    args = ap.parse_args()

    # carregar
    df = pd.read_csv(args.src).reset_index(drop=True)
    n_all = len(df)

    # colunas base
    tname_col = find_col(df, ["tournament","tournament_name","event","event_name","competition","league","competition_name"])
    level_col = find_col(df, ["category","level","series","tier","tournament_level","tournament_category","event_level"])
    tour_col  = find_col(df, ["tour","tour_name","gender","circuit"])
    bo_col    = find_col(df, ["best_of","bo","sets","format"])
    round_col = find_col(df, ["round","stage","phase"])

    tname = df[tname_col].astype(str) if tname_col else pd.Series([""]*n_all)
    level = df[level_col].astype(str) if level_col else pd.Series([""]*n_all)
    tourv = df[tour_col].astype(str) if tour_col else pd.Series([""]*n_all)
    round_txt = df[round_col].astype(str) if round_col else pd.Series([""]*n_all)
    bo_num = pd.to_numeric(df[bo_col], errors="coerce").fillna(0).astype(int) if bo_col else pd.Series([0]*n_all)

    # categorias/tour + inferências
    cat_tag  = [detect_category(tname.iloc[i], level.iloc[i]) for i in range(n_all)]
    bestof_guess = [infer_best_of(cat_tag[i], int(bo_num.iloc[i]), round_txt.iloc[i]) for i in range(n_all)]
    tour_tag = [adjust_tour_if_gs(
        detect_tour_from_text(tname.iloc[i], tourv.iloc[i]),
        cat_tag[i], bestof_guess[i]
    ) for i in range(n_all)]

    # máscara de filtros
    mask = np.ones(n_all, dtype=bool)

    if args.tour != "both":
        mask &= (np.array(tour_tag) == args.tour)

    cats_req = set()
    if args.categories:
        for tok in re.split(r"[,\s]+", args.categories.lower()):
            if not tok: continue
            if tok in ("gs","grand_slam","slam"): tok = "gs"
            if tok in ("m1000","masters","masters1000","atp1000"): tok = "1000"
            cats_req.add(tok)
    if cats_req:
        mask &= np.array([c in cats_req for c in cat_tag])

    if args.best_of in (3,5):
        mask &= (np.array(bestof_guess) == int(args.best_of))

    if args.name_like and tname_col:
        try:
            rx = re.compile(args.name_like, re.IGNORECASE)
            mask &= tname.apply(lambda s: bool(rx.search(s))).to_numpy()
        except re.error:
            print(f"[totals] aviso: regex inválido em --name-like: {args.name_like}")

    # aplicar filtros
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        pd.DataFrame().to_csv(args.out, index=False)
        cats_repr = sorted(list(cats_req)) if cats_req else "all"
        print(f"[totals] 0 jogos após filtros (tour={args.tour}, cats={cats_repr}, bo={args.best_of or 'all'}) -> {args.out}")
        return

    # Recalcular meta ALINHADA ao DF filtrado (evita 'tournament' repetido)
    n0 = len(df)
    tname = df[tname_col].astype(str) if tname_col else pd.Series([""]*n0)
    level = df[level_col].astype(str) if level_col else pd.Series([""]*n0)
    tourv = df[tour_col].astype(str) if tour_col else pd.Series([""]*n0)
    round_txt = df[round_col].astype(str) if round_col else pd.Series([""]*n0)
    bo_num = pd.to_numeric(df[bo_col], errors="coerce").fillna(0).astype(int) if bo_col else pd.Series([0]*n0)

    cat_tag  = [detect_category(tname.iloc[i], level.iloc[i]) for i in range(n0)]
    bestof_guess = [infer_best_of(cat_tag[i], int(bo_num.iloc[i]), round_txt.iloc[i]) for i in range(n0)]
    tour_tag = [adjust_tour_if_gs(
        detect_tour_from_text(tname.iloc[i], tourv.iloc[i]),
        cat_tag[i], bestof_guess[i]
    ) for i in range(n0)]

    # probabilidades
    p1c = find_col(df, ["p1_prob_adj","p1_prob","prob_p1","proba_p1"])
    p2c = find_col(df, ["p2_prob_adj","p2_prob","prob_p2","proba_p2"])
    pred = find_col(df, ["pred_prob","proba","p_win","win_prob"])
    if p1c is None or p2c is None:
        if pred is None:
            raise SystemExit("Não encontrei probabilidades: precisa de p1/p2 ou pred_prob.")
        df["p1_prob_tmp"] = pd.to_numeric(df[pred], errors="coerce").fillna(0.5).clip(0,1)
        df["p2_prob_tmp"] = 1.0 - df["p1_prob_tmp"]
        p1c, p2c = "p1_prob_tmp", "p2_prob_tmp"
    df[p1c] = pd.to_numeric(df[p1c], errors="coerce").fillna(0.5).clip(0,1)
    df[p2c] = pd.to_numeric(df[p2c], errors="coerce").fillna(0.5).clip(0,1)

    # jogadores/datas
    p1n = find_col(df, ["player1","p1","home"])
    p2n = find_col(df, ["player2","p2","away"])
    dcol = find_col(df, ["start_time","match_date","date","start","kickoff"])
    match_dates = pd.to_datetime(df[dcol], errors="coerce", utc=True) if dcol else pd.Series([pd.NaT]*n0)

    # surface
    sfc_col = None
    for c in df.columns:
        if "surface" in c.lower():
            sfc_col = c
            break
    if sfc_col is None:
        sfc_col = find_col(df, ["surface","court","surface_name"])

    def row_surface(i: int) -> str:
        if sfc_col:
            return parse_surface(df.iloc[i][sfc_col])
        return "hard"

    # notícias
    news_df = None
    if args.news:
        try:
            news_df = load_news(args.news)
        except Exception as e:
            print(f"[totals] aviso: falha a ler notícias: {e}")

    # competitividade (0..1) = 2*min(p1,p2)
    comp = (2.0 * np.minimum(df[p1c].to_numpy(dtype=float), df[p2c].to_numpy(dtype=float))).clip(0,1)

    # riscos por notícias
    r1 = np.zeros(n0); r2 = np.zeros(n0)
    if news_df is not None and not news_df.empty and p1n and p2n:
        for i in range(n0):
            dt = match_dates.iloc[i].to_pydatetime() if (dcol and not pd.isna(match_dates.iloc[i])) else None
            r1[i] = player_risk(str(df.iloc[i][p1n]), dt, news_df, args.half_life)
            r2[i] = player_risk(str(df.iloc[i][p2n]), dt, news_df, args.half_life)
    risk_max = np.maximum(r1, r2)

    # média esperada
    mu_map = {"hard": args.mu_hard, "clay": args.mu_clay, "grass": args.mu_grass, "carpet": args.mu_carpet}
    mu = np.zeros(n0, dtype=float)
    for i in range(n0):
        base = mu_map.get(row_surface(i), args.mu_hard)
        shift_comp = args.comp_gain * (comp[i] - 0.5)
        shift_news = - args.news_gamma * risk_max[i]
        bonus_bo5 = args.bo5_bonus if bestof_guess[i] >= 5 else 0.0
        mu[i] = base + shift_comp + shift_news + bonus_bo5
    mu = np.clip(mu, 16.0, 45.0)
    sigma = float(args.sigma)

    # linhas alvo
    if args.line is not None:
        lines_all = [float(args.line)]
    else:
        # aceita "24,5" ou "24.5"
        tokens = [t for t in re.split(r"[,\s]+", str(args.lines)) if t.strip()]
        lines_all = [float(t.replace(",", ".")) for t in tokens]

    # gerar picks
    out_rows = []
    for i in range(n0):
        for L in lines_all:
            z_over = (L + 0.5 - mu[i]) / sigma
            p_over = 1.0 - norm_cdf(z_over)
            z_under = (L - 0.5 - mu[i]) / sigma
            p_under = norm_cdf(z_under)

            def add(side: str, prob: float):
                row = {
                    "line": L,
                    "side": side.upper(),
                    "prob": round(float(prob), 4),
                    "mu": round(float(mu[i]), 3),
                    "sigma": sigma,
                    "surface": row_surface(i),
                    "best_of": int(bestof_guess[i]),
                    "tour": tour_tag[i],
                    "category": cat_tag[i],
                    "tournament": tname.iloc[i] if len(tname) > i else "",
                    "p1_prob": float(df.iloc[i][p1c]),
                    "p2_prob": float(df.iloc[i][p2c]),
                    "competitiveness": float(comp[i]),
                    "news_risk_max": float(risk_max[i]),
                }
                if p1n: row["player1"] = str(df.iloc[i][p1n])
                if p2n: row["player2"] = str(df.iloc[i][p2n])
                for k in ["round","match_id","event","start_time","date"]:
                    if k in df.columns:
                        row[k] = df.iloc[i][k]
                out_rows.append(row)

            if args.side in ("both","over") and p_over >= args.min_prob:
                add("OVER", p_over)
            if args.side in ("both","under") and p_under >= args.min_prob:
                add("UNDER", p_under)

    out_df = pd.DataFrame(out_rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["prob","line"], ascending=[False, True])

    out_df.to_csv(args.out, index=False)
    cats_repr = args.categories or "all"
    print(f"[totals] {n0} jogos após filtros (tour={args.tour}, cats={cats_repr}, bo={args.best_of or 'all'}) • linhas={lines_all} • side={args.side} • min_prob={args.min_prob:.2f} -> {len(out_df)} picks -> {args.out}")

if __name__ == "__main__":
    main()
