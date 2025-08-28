
import pandas as pd

def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date","player1","player2","winner","surface","level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no CSV de históricos: {missing}")
    return df

def load_fixtures(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"player1","player2","surface","level","odds_p1","odds_p2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no CSV de fixtures: {missing}")
    return df
