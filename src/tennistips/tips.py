import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

# --- Fatigue: cria features e ajusta probabilidades ---
from .features.fatigue import (
    add_fatigue_features,
    adjust_probs_for_fixtures_row,
    FatigueParams,
)

# pesos/base defaults para o ajuste (podes afinar no ficheiro de fatigue)
FATIGUE = FatigueParams()


# ------------------ util ------------------

def _norm_name(s: str) -> str:
    return str(s).strip()

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


# ------------------ estado por jogador (ELO/H2H/forma) ------------------

SURFACES = ("Hard", "Clay", "Grass", "Carpet")

@dataclass
class PlayerState:
    elo_all: float
    elo_surface: Dict[str, float] = field(default_factory=lambda: {s: 1500.0 for s in SURFACES})
    matches_played: int = 0
    recent_results: List[int] = field(default_factory=list)  # 1 win / 0 loss
    last_match_date: datetime | None = None
    h2h: Dict[str, Tuple[int, int, float]] = field(default_factory=dict)  # opp -> (wins, losses, score)

def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))

def _update_elo(r_a: float, r_b: float, score_a: float, k: float) -> Tuple[float, float]:
    ea = _expected(r_a, r_b)
    eb = 1.0 - ea
    r_a2 = r_a + k * (score_a - ea)
    r_b2 = r_b + k * ((1.0 - score_a) - eb)
    return r_a2, r_b2

def build_player_states(
    history_df: pd.DataFrame,
    elo_start: float,
    k_base: float,
    surface_k_boost: float,
    form_window: int,
    h2h_decay: float,
) -> Dict[str, PlayerState]:
    """Varre o histórico em ordem cronológica e constrói estado por jogador."""
    df = history_df.copy()
    # colunas exigidas
    req = {"date", "player1", "player2", "winner", "surface", "level"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Histórico sem colunas necessárias: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    players: Dict[str, PlayerState] = {}

    def get_state(name: str) -> PlayerState:
        name = _norm_name(name)
        if name not in players:
            st = PlayerState(elo_all=elo_start,
                             elo_surface={s: elo_start for s in SURFACES},
                             matches_played=0,
                             recent_results=[],
                             last_match_date=None,
                             h2h={})
            players[name] = st
        return players[name]

    for _, r in df.iterrows():
        p1 = _norm_name(r["player1"])
        p2 = _norm_name(r["player2"])
        w  = _norm_name(r["winner"])
        surf = str(r["surface"]) if str(r["surface"]) in SURFACES else "Hard"
        d = pd.to_datetime(r["date"])

        s1 = get_state(p1)
        s2 = get_state(p2)

        # K da partida (boost pequeno por superfície)
        k = k_base * (surface_k_boost if surf in SURFACES else 1.0)

        # resultado do ponto de vista do player1
        score1 = 1.0 if p1 == w else 0.0
        score2 = 1.0 - score1

        # update ELO global
        s1.elo_all, s2.elo_all = _update_elo(s1.elo_all, s2.elo_all, score1, k)
        # update ELO por superfície
        e1, e2 = s1.elo_surface[surf], s2.elo_surface[surf]
        e1n, e2n = _update_elo(e1, e2, score1, k)
        s1.elo_surface[surf], s2.elo_surface[surf] = e1n, e2n

        # forma recente (janela)
        s1.recent_results.append(int(score1))
        s2.recent_results.append(int(score2))
        if len(s1.recent_results) > form_window:
            s1.recent_results = s1.recent_results[-form_window:]
        if len(s2.recent_results) > form_window:
            s2.recent_results = s2.recent_results[-form_window:]

        # h2h decaído
        w1, l1, sc1 = s1.h2h.get(p2, (0, 0, 0.0))
        w2, l2, sc2 = s2.h2h.get(p1, (0, 0, 0.0))
        sc1 = sc1 * h2h_decay + (1.0 if score1 == 1.0 else -1.0)
        sc2 = sc2 * h2h_decay + (1.0 if score2 == 1.0 else -1.0)
        s1.h2h[p2] = (w1 + (1 if score1 == 1.0 else 0), l1 + (1 if score1 == 0.0 else 0), sc1)
        s2.h2h[p1] = (w2 + (1 if score2 == 1.0 else 0), l2 + (1 if score2 == 0.0 else 0), sc2)

        # housekeeping
        s1.matches_played += 1
        s2.matches_played += 1
        s1.last_match_date = d
        s2.last_match_date = d

    return players


def match_features_from_state(
    players: Dict[str, PlayerState],
    p1: str, p2: str, surface: str,
    elo_start: float,
    form_window: int,
    h2h_decay: float,   # não usamos diretamente aqui (já aplicado ao construir o estado)
) -> np.ndarray:
    """Extrai o vetor (1,4) [elo_diff, elo_surf_diff, form_diff, h2h_diff] para p1 vs p2."""
    p1 = _norm_name(p1)
    p2 = _norm_name(p2)
    surf = surface if surface in SURFACES else "Hard"

    # init se jogador novo
    if p1 not in players:
        players[p1] = PlayerState(elo_all=elo_start,
                                  elo_surface={s: elo_start for s in SURFACES},
                                  recent_results=[],
                                  h2h={})
    if p2 not in players:
        players[p2] = PlayerState(elo_all=elo_start,
                                  elo_surface={s: elo_start for s in SURFACES},
                                  recent_results=[],
                                  h2h={})

    s1 = players[p1]
    s2 = players[p2]

    elo_diff = s1.elo_all - s2.elo_all
    elo_surf_diff = s1.elo_surface.get(surf, elo_start) - s2.elo_surface.get(surf, elo_start)

    # forma = média dos últimos resultados
    f1 = np.mean(s1.recent_results) if len(s1.recent_results) else 0.0
    f2 = np.mean(s2.recent_results) if len(s2.recent_results) else 0.0
    form_diff = float(f1 - f2)

    # h2h: usa o score decaído
    h1 = s1.h2h.get(p2, (0, 0, 0.0))[2]
    h2 = s2.h2h.get(p1, (0, 0, 0.0))[2]
    h2h_diff = float(h1 - h2)

    return np.array([[elo_diff, elo_surf_diff, form_diff, h2h_diff]], dtype=float)


# ------------------ geração de tips ------------------

def _kelly_fraction(p: float, odds: float) -> float:
    # Kelly para prob p e odd decimal
    b = odds - 1.0
    return max(0.0, (b * p - (1.0 - p)) / b) if b > 0 else 0.0

def generate_tips(history_df: pd.DataFrame, fixtures_df: pd.DataFrame, model, cfg) -> pd.DataFrame:
    # odds válidas (>1)
    if "odds_p1" in fixtures_df.columns and "odds_p2" in fixtures_df.columns:
        fixtures_df = fixtures_df[(fixtures_df["odds_p1"] > 1) & (fixtures_df["odds_p2"] > 1)]

    # datas futuras e após o histórico
    if "date" in fixtures_df.columns:
        today = datetime.now(timezone.utc).date()
        fixtures_df = fixtures_df.copy()
        fixtures_df["date"] = pd.to_datetime(fixtures_df["date"], errors="coerce").dt.date

        # histórico já deve estar com date datetime64
        hist_dates = pd.to_datetime(history_df["date"], errors="coerce")
        last_hist_date = hist_dates.max().date()

        fixtures_df = fixtures_df[fixtures_df["date"].notna()]
        fixtures_df = fixtures_df[fixtures_df["date"] >= today]
        fixtures_df = fixtures_df[fixtures_df["date"] > last_hist_date]

    # se esvaziou, retorna schema vazio
    if fixtures_df.empty:
        return pd.DataFrame(columns=[
            "player1","player2","surface","odds_p1","odds_p2",
            "p1_prob","p2_prob","ev_p1","ev_p2","pick","best_ev","stake_suggest"
        ])

    # --- Garante features de fadiga no fixtures_df (gera se faltarem) ---
    need_cols = {
        "p1_matches_7d","p1_matches_14d","p1_matches_30d","p1_b2b","p1_rest_48h",
        "p2_matches_7d","p2_matches_14d","p2_matches_30d","p2_b2b","p2_rest_48h",
    }
    if not need_cols.issubset(set(fixtures_df.columns)):
        fixtures_df = add_fatigue_features(fixtures_df, history_df)

    # construir estado a partir do histórico
    players = build_player_states(
        history_df,
        cfg.elo.start,
        cfg.elo.k_base,
        cfg.elo.surface_k_boost,
        cfg.features.form_window,
        cfg.features.h2h_decay,
    )

    rows = []
    for _, r in fixtures_df.iterrows():
        p1 = _norm_name(r["player1"])
        p2 = _norm_name(r["player2"])
        surface = str(r.get("surface", "Hard"))
        o1 = _safe_float(r.get("odds_p1"), np.nan)
        o2 = _safe_float(r.get("odds_p2"), np.nan)

        # vetor de features
        X = match_features_from_state(
            players, p1, p2, surface,
            cfg.elo.start, cfg.features.form_window, cfg.features.h2h_decay
        )

        # prob do modelo (wrapper retorna (n,2))
        p = model.predict_proba(X)
        p1_win = float(p[:, 1][0])
        p2_win = 1.0 - p1_win

        # --- AJUSTE por FADIGA (penaliza quem está mais “carregado”) ---
        p1_win, p2_win = adjust_probs_for_fixtures_row(r, p1_win, p2_win, FATIGUE)

        # EV simples (decimais): EV = p*odds - (1-p)
        ev1 = p1_win * o1 - (1.0 - p1_win) if o1 == o1 else np.nan
        ev2 = p2_win * o2 - (1.0 - p2_win) if o2 == o2 else np.nan

        # escolha pela maior EV
        if (ev1 if np.isfinite(ev1) else -1e9) >= (ev2 if np.isfinite(ev2) else -1e9):
            pick = "P1"
            best_ev = ev1
            stake = _kelly_fraction(p1_win, o1) * getattr(cfg.selection, "kelly_fraction", 0.25)
        else:
            pick = "P2"
            best_ev = ev2
            stake = _kelly_fraction(p2_win, o2) * getattr(cfg.selection, "kelly_fraction", 0.25)

        # aplica cap e threshold de EV
        cap = getattr(cfg.selection, "kelly_cap", 0.05)
        stake = min(float(stake), float(cap))

        rows.append({
            "player1": p1,
            "player2": p2,
            "surface": surface,
            "odds_p1": o1,
            "odds_p2": o2,
            "p1_prob": round(p1_win, 4),
            "p2_prob": round(p2_win, 4),
            "ev_p1": round(ev1, 4) if np.isfinite(ev1) else np.nan,
            "ev_p2": round(ev2, 4) if np.isfinite(ev2) else np.nan,
            "pick": pick,
            "best_ev": round(best_ev, 4) if np.isfinite(best_ev) else np.nan,
            "stake_suggest": round(stake, 4),
        })

    tips = pd.DataFrame(rows)

    # filtra por threshold de EV se definido
    thr = getattr(cfg.selection, "ev_threshold", 0.0)
    tips = tips[(tips["best_ev"].astype(float) >= float(thr)) | (~np.isfinite(tips["best_ev"].astype(float)))]

    # ordena por EV desc
    tips = tips.sort_values("best_ev", ascending=False, na_position="last").reset_index(drop=True)
    return tips
