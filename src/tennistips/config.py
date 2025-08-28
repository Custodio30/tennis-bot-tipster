
from dataclasses import dataclass
from typing import Literal
import yaml

@dataclass
class EloCfg:
    start: float
    k_base: float
    surface_k_boost: float

@dataclass
class FeatCfg:
    form_window: int
    h2h_decay: float

@dataclass
class ModelCfg:
    type: Literal["logreg"]
    calibration: Literal["sigmoid","isotonic"]
    test_size: float
    shuffle: bool
    max_iter: int

@dataclass
class SelectionCfg:
    ev_threshold: float
    kelly_fraction: float
    kelly_cap: float

@dataclass
class Cfg:
    seed: int
    elo: EloCfg
    features: FeatCfg
    model: ModelCfg
    selection: SelectionCfg

def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Cfg(
        seed = d["seed"],
        elo = EloCfg(**d["elo"]),
        features = FeatCfg(**d["features"]),
        model = ModelCfg(**d["model"]),
        selection = SelectionCfg(**d["selection"]),
    )
