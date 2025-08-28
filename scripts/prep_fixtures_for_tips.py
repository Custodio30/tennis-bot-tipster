# scripts/prep_fixtures_for_tips.py  (substitui o conteúdo anterior)
import pandas as pd
import sys, os

src = sys.argv[1] if len(sys.argv) > 1 else "data/fixtures/latest.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "data/fixtures/latest_for_tips.csv"

df = pd.read_csv(src)

# Renomear para o que o tips.py espera
df = df.rename(columns={"home": "player1", "away": "player2"})

# Odds: se não existirem ou vierem vazias, mete padrão 1.90
for c in ["odds_p1", "odds_p2"]:
    if c not in df.columns:
        df[c] = 1.90
df["odds_p1"] = df["odds_p1"].fillna(1.90)
df["odds_p2"] = df["odds_p2"].fillna(1.90)

# Surface default
if "surface" in df.columns:
    df["surface"] = df["surface"].fillna("Hard")
else:
    df["surface"] = "Hard"

os.makedirs(os.path.dirname(dst), exist_ok=True)
df.to_csv(dst, index=False)
print(f"[ok] Fixtures preparados → {dst} ({len(df)} linhas)")
