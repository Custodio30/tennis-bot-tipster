
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np

@dataclass
class PlayerState:
    elo_all: float
    elo_surface: Dict[str, float]
    matches_played: int
    recent_results: List[int]
    last_match_date: Optional[pd.Timestamp]
    h2h: Dict[str, Tuple[int,int,float]]

def new_player(start_elo: float) -> PlayerState:
    return PlayerState(
        elo_all=start_elo,
        elo_surface={},
        matches_played=0,
        recent_results=[],
        last_match_date=None,
        h2h={},
    )

def get_surface_elo(st: PlayerState, surface: str, start: float) -> float:
    return st.elo_surface.get(surface, start)

def update_surface_elo(st: PlayerState, surface: str, new_elo: float):
    st.elo_surface[surface] = new_elo

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a)/400.0))

def k_factor(k_base: float, level: str, matches_played: int, surface_specific: bool, surface_k_boost: float) -> float:
    k = k_base
    if level and level.upper() in ("ATP","WTA"):
        k *= 1.0
    if matches_played < 30:
        k *= 1.10
    if surface_specific:
        k *= surface_k_boost
    return k
