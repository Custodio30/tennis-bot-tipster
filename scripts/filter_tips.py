# scripts/filter_tips.py
import sys, os
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "outputs/tips.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "outputs/tips_filtered.csv"

df = pd.read_csv(src)

# 1) remover pares/doubles (heurística: tem " / " no nome)
is_doubles = df["player1"].astype(str).str.contains(r"\s/\s") | df["player2"].astype(str).str.contains(r"\s/\s")
df = df[~is_doubles].copy()

# 2) focar níveis principais (ajusta se quiseres incluir ITF)
keep_levels = ["ATP", "WTA", "Challenger"]
if "category" in df.columns:
    df = df[df["category"].isin(keep_levels)]

# 3) aplicar thresholds de qualidade
# - valor esperado mínimo
MIN_EV = 0.05
# - prob mínima no lado escolhido (evita picks 50/50)
MIN_PROB = 0.60

def prob_of_pick(row):
    return row["p1_prob"] if row["pick"] == "P1" else row["p2_prob"]

def ev_of_pick(row):
    return row["ev_p1"] if row["pick"] == "P1" else row["ev_p2"]

df["pick_prob"] = df.apply(prob_of_pick, axis=1)
df["pick_ev"] = df.apply(ev_of_pick, axis=1)

df = df[(df["pick_ev"] >= MIN_EV) & (df["pick_prob"] >= MIN_PROB)].copy()

# 4) ordenar por EV e limitar top N se quiseres
TOP_N = 30
df = df.sort_values(["pick_ev", "pick_prob"], ascending=False).head(TOP_N)

# 5) colunas úteis
cols = ["player1","player2","category","surface","odds_p1","odds_p2",
        "p1_prob","p2_prob","pick","pick_ev","stake_suggest"]
df = df[[c for c in cols if c in df.columns]]

os.makedirs(os.path.dirname(dst), exist_ok=True)
df.to_csv(dst, index=False)
print(f"[ok] Salvo {len(df)} picks → {dst}")
