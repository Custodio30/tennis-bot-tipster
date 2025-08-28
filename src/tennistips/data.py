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
    df = _read_csv_any(path)

    # garantir coluna de data como datetime (mesmo que já venha)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    else:
        raise ValueError("Falta a coluna 'date' no CSV de históricos.")

    # Se não houver winner (fallback), cria a partir de player1
    if "winner" not in df.columns and {"player1", "player2"}.issubset(df.columns):
        df["winner"] = df["player1"]

    # normalizações leves
    for col in ["player1", "player2", "winner", "surface", "level"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # validação final
    missing = _REQUIRED_MATCH_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no CSV de históricos: {missing}")

    # opcional: ordenar por data
    df = df.sort_values("date").reset_index(drop=True)
    return df

def load_fixtures(path: str) -> pd.DataFrame:
    df = _read_csv_any(path)

    # requisitos mínimos para prever
    needed = {"date", "player1", "player2"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no CSV de fixtures: {missing} (mínimo: {sorted(needed)})")

    # datas
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # defaults úteis (caso não venham no ficheiro)
    if "surface" not in df.columns:
        df["surface"] = "Unknown"
    if "level" not in df.columns:
        df["level"] = "ATP"  # o teu k_factor já tolera

    for col in ["player1", "player2", "surface", "level"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.sort_values("date").reset_index(drop=True)
    return df
