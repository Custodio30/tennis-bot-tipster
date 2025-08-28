import pandas as pd

# Colunas esperadas para históricos (matches)
_REQUIRED_MATCH_COLS = {"date", "player1", "player2", "winner", "surface", "level"}

def _read_csv_any(path: str) -> pd.DataFrame:
    """Tenta ler CSV com UTF-8 e, se houver BOM, usa utf-8-sig."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")

def load_matches(path: str) -> pd.DataFrame:
    # lê csv e já tenta converter "date"
    df = _read_csv_any(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # se não houver winner (fallback), assume player1 como winner
    if "winner" not in df.columns and {"player1", "player2"}.issubset(df.columns):
        df["winner"] = df["player1"]

    # checa colunas obrigatórias
    missing = _REQUIRED_MATCH_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no CSV de históricos: {missing}")

    return df

def load_fixtures(path: str) -> pd.DataFrame:
    df = _read_csv_any(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
