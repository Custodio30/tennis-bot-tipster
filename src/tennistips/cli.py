import typer
import pandas as pd
from typing import List

from .config import load_config
from .data import load_matches, load_fixtures
from .features import build_features
from .model import (
    train_logreg_calibrated,         # legado (holdout simples)
    train_logreg_ts_cv,              # novo: CV temporal + calibração
    train_hgb_ts_cv,                 # novo: Gradient Boosting + calibração
    save_model, load_model
)
from .tips import generate_tips
from .news_guard import build_news_flags, _norm_name

from .pipeline.downloader import fetch_sackmann, fetch_tennis_data
from .pipeline.merge import build_merged_dataset

app = typer.Typer(add_completion=False, help="🎾 TennisTips CLI")


# ---------- helper: duplicar e espelhar amostras p/ evitar classe única ----------
def _mirror_features(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicamos as amostras trocando virtualmente p1<->p2:
    - inverte sinais das features diferenciais
    - inverte o rótulo (y -> 1 - y)
    Isto equilibra o treino quando o histórico vem com player1==winner.
    """
    mir = feats.copy()
    mir["y"] = 1 - mir["y"]
    for col in ["elo_diff", "elo_surf_diff", "form_diff", "h2h_diff"]:
        mir[col] = -mir[col]
    return pd.concat([feats, mir], ignore_index=True)


@app.command()
def train(
    history: str = typer.Option(..., help="CSV de históricos"),
    config: str = typer.Option("configs/default.yaml", help="Caminho para YAML de config"),
    model_path: str = typer.Option("models/model.joblib", help="Destino para gravar o modelo")
):
    """
    Treina um modelo conforme 'model.type' no YAML: 'logreg' (default), 'hgb' ou 'ensemble'.
    Todos são guardados com uma interface compatível (predict_proba).
    """
    cfg = load_config(config)
    df = load_matches(history)
    feats = build_features(
        df, cfg.elo.start, cfg.elo.k_base, cfg.elo.surface_k_boost,
        cfg.features.form_window, cfg.features.h2h_decay
    )
    # <<< espelhamento para evitar classe única >>>
    feats = _mirror_features(feats)

    X = feats[["elo_diff", "elo_surf_diff", "form_diff", "h2h_diff"]].values
    y = feats["y"].values

    mtype = getattr(cfg.model, "type", "logreg")

    if mtype == "logreg":
        out = train_logreg_ts_cv(
            X, y,
            max_iter=getattr(cfg.model, "max_iter", 2000),
            calibration=getattr(cfg.model, "calibration", "isotonic"),
            n_splits=getattr(cfg.model, "n_splits", 5),
            seed=cfg.seed
        )
        model = out["model"]
        typer.echo(f"[logreg-tscv] CV AUC={out['cv_auc']:.4f} | CV LogLoss={out['cv_logloss']:.4f}")

    elif mtype == "hgb":
        out = train_hgb_ts_cv(
            X, y,
            n_splits=getattr(cfg.model, "n_splits", 5),
            seed=cfg.seed,
            max_depth=getattr(cfg.model, "max_depth", 6),
            learning_rate=getattr(cfg.model, "learning_rate", 0.06),
            max_iter=getattr(cfg.model, "max_iter", 400)
        )
        model = out["model"]
        typer.echo(f"[hgb-tscv] CV AUC={out['cv_auc']:.4f} | CV LogLoss={out['cv_logloss']:.4f}")

    elif mtype == "ensemble":
        # Treina ambos e faz média das probabilidades (peso configurável)
        out_lr = train_logreg_ts_cv(
            X, y,
            max_iter=getattr(cfg.model, "max_iter", 2000),
            calibration=getattr(cfg.model, "calibration", "isotonic"),
            n_splits=getattr(cfg.model, "n_splits", 5),
            seed=cfg.seed
        )
        out_hgb = train_hgb_ts_cv(
            X, y,
            n_splits=getattr(cfg.model, "n_splits", 5),
            seed=cfg.seed,
            max_depth=getattr(cfg.model, "max_depth", 6),
            learning_rate=getattr(cfg.model, "learning_rate", 0.06),
            max_iter=getattr(cfg.model, "max_iter", 400)
        )
        model = ("ensemble", {"lr": out_lr["model"], "hgb": out_hgb["model"], "w": getattr(cfg.model, "ensemble_weight", 0.5)})
        typer.echo(f"[ensemble] ~AUC={(out_lr['cv_auc'] + out_hgb['cv_auc'])/2:.4f} | ~LL={(out_lr['cv_logloss'] + out_hgb['cv_logloss'])/2:.4f}")
    else:
        # fallback para a função antiga (holdout simples)
        model, metrics = train_logreg_calibrated(
            X, y,
            getattr(cfg.model, "test_size", 0.2),
            getattr(cfg.model, "shuffle", False),
            getattr(cfg.model, "max_iter", 1000),
            getattr(cfg.model, "calibration", "sigmoid"),
            cfg.seed
        )
        typer.echo(f"[logreg-legacy] log_loss={metrics['log_loss']:.4f}, AUC={metrics['auc']:.4f}")

    save_model(model, model_path)
    typer.echo(f"Modelo guardado em: {model_path}")


@app.command()
def tips(
    history: str = typer.Option(..., help="CSV de históricos"),
    fixtures: str = typer.Option(..., help="CSV de fixtures"),
    config: str = typer.Option("configs/default.yaml", help="Config YAML"),
    model_path: str = typer.Option("models/model.joblib", help="Modelo treinado (.joblib)"),
    out: str = typer.Option("outputs/tips.csv", help="Caminho CSV de saída")
):
    cfg = load_config(config)
    hist = load_matches(history)
    fx = load_fixtures(fixtures)
    model = load_model(model_path)  # devolve um objeto com predict_proba()
    tips_df = generate_tips(hist, fx, model, cfg)

    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tips_df.to_csv(out, index=False)
    typer.echo(f"Gerado {len(tips_df)} tips → {out}")
    if len(tips_df) > 0:
        typer.echo(tips_df.to_string(index=False))


@app.command()
def eval(
    history: str = typer.Option(..., help="CSV de históricos"),
    config: str = typer.Option("configs/default.yaml", help="Config YAML")
):
    """
    Avaliação rápida com TimeSeriesSplit para ver AUC/logloss médios.
    """
    cfg = load_config(config)
    df = load_matches(history)
    feats = build_features(
        df, cfg.elo.start, cfg.elo.k_base, cfg.elo.surface_k_boost,
        cfg.features.form_window, cfg.features.h2h_decay
    )
    # <<< espelhamento para avaliação coerente >>>
    feats = _mirror_features(feats)

    X = feats[["elo_diff", "elo_surf_diff", "form_diff", "h2h_diff"]].values
    y = feats["y"].values

    mtype = getattr(cfg.model, "type", "logreg")
    if mtype == "logreg":
        out = train_logreg_ts_cv(X, y, max_iter=getattr(cfg.model, "max_iter", 2000),
                                 calibration=getattr(cfg.model, "calibration", "isotonic"),
                                 n_splits=getattr(cfg.model, "n_splits", 5), seed=cfg.seed)
        typer.echo(f"Eval[logreg-tscv] CV AUC={out['cv_auc']:.4f} | CV LogLoss={out['cv_logloss']:.4f}")
    elif mtype == "hgb":
        out = train_hgb_ts_cv(X, y, n_splits=getattr(cfg.model, "n_splits", 5),
                              seed=cfg.seed,
                              max_depth=getattr(cfg.model, "max_depth", 6),
                              learning_rate=getattr(cfg.model, "learning_rate", 0.06),
                              max_iter=getattr(cfg.model, "max_iter", 400))
        typer.echo(f"Eval[hgb-tscv] CV AUC={out['cv_auc']:.4f} | CV LogLoss={out['cv_logloss']:.4f}")
    else:
        out_lr = train_logreg_ts_cv(X, y, max_iter=getattr(cfg.model, "max_iter", 2000),
                                    calibration=getattr(cfg.model, "calibration", "isotonic"),
                                    n_splits=getattr(cfg.model, "n_splits", 5), seed=cfg.seed)
        out_hgb = train_hgb_ts_cv(X, y, n_splits=getattr(cfg.model, "n_splits", 5),
                                  seed=cfg.seed,
                                  max_depth=getattr(cfg.model, "max_depth", 6),
                                  learning_rate=getattr(cfg.model, "learning_rate", 0.06),
                                  max_iter=getattr(cfg.model, "max_iter", 400))
        typer.echo(f"Eval[ensemble] ~AUC={(out_lr['cv_auc'] + out_hgb['cv_auc'])/2:.4f} | ~LL={(out_lr['cv_logloss'] + out_hgb['cv_logloss'])/2:.4f}")


@app.command()
def news(
    fixtures: str = typer.Option(..., help="CSV de fixtures para verificar"),
    extra_feed: List[str] = typer.Option(None, help="RSS extra")
):
    """
    Verifica jogadores dos fixtures contra feeds de notícias/lesões
    e mostra quem deve ser bloqueado/penalizado.
    """
    import json
    fx = pd.read_csv(fixtures)
    player_index = set(_norm_name(p) for p in pd.concat([fx["player1"], fx["player2"]]).dropna().astype(str))
    flags = build_news_flags(player_index, extra_feeds=extra_feed or [])
    typer.echo(json.dumps(flags, indent=2, ensure_ascii=False))


@app.command()
def fetch(
    config: str = typer.Option("configs/data_sources.yaml", help="Config de fontes de dados"),
    sources: str = typer.Option("both", help="sackmann|tennisdata|both")
):
    if sources in ("sackmann", "both"):
        files = fetch_sackmann(config)
        typer.echo(f"Descargados {len(files)} ficheiros Sackmann para data/raw/")
    if sources in ("tennisdata", "both"):
        files2 = fetch_tennis_data(config)
        typer.echo(f"Descargados {len(files2)} ficheiros Tennis-Data para data/raw/")
    typer.echo("✅ Fetch completo.")


@app.command()
def builddataset(
    config: str = typer.Option("configs/data_sources.yaml", help="Config de fontes de dados")
):
    out = build_merged_dataset(config)
    typer.echo(f"✅ Dataset consolidado criado em: {out}")


if __name__ == "__main__":
    app()
