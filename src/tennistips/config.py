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
    type: Literal["logreg", "hgb", "ensemble"] = "logreg"
    calibration: Literal["sigmoid","isotonic"] = "isotonic"
    test_size: float = 0.2
    shuffle: bool = False
    max_iter: int = 2000
    # novos (para CV temporal / HGB / ensemble)
    n_splits: int = 5
    max_depth: int = 6
    learning_rate: float = 0.06
    ensemble_weight: float = 0.5

@dataclass
class SelectionCfg:
    ev_threshold: float
    kelly_fraction: float
    kelly_cap: float  # <- garantir float

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
