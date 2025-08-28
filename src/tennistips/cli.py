
import typer
import pandas as pd
from .config import load_config
from .data import load_matches, load_fixtures
from .features import build_features
from .model import train_logreg_calibrated, save_model, load_model
from .tips import generate_tips

app = typer.Typer(add_completion=False, help="🎾 TennisTips CLI")

from .pipeline.downloader import fetch_sackmann, fetch_tennis_data
from .pipeline.merge import build_merged_dataset

@app.command()
def train(history: str = typer.Option(..., help="CSV de históricos"),
          config: str = typer.Option("configs/default.yaml", help="Caminho para YAML de config"),
          model_path: str = typer.Option("models/model.joblib", help="Destino para gravar o modelo")):
    cfg = load_config(config)
    df = load_matches(history)
    feats = build_features(df, cfg.elo.start, cfg.elo.k_base, cfg.elo.surface_k_boost, cfg.features.form_window, cfg.features.h2h_decay)
    X = feats[["elo_diff","elo_surf_diff","form_diff","h2h_diff"]].values
    y = feats["y"].values
    model, metrics = train_logreg_calibrated(X, y, cfg.model.test_size, cfg.model.shuffle, cfg.model.max_iter, cfg.model.calibration, cfg.seed)
    save_model(model, model_path)
    typer.echo(f"Treino concluído: log_loss={metrics['log_loss']:.4f}, AUC={metrics['auc']:.4f} "
               f"(train={metrics['samples_train']}, val={metrics['samples_val']})")
    typer.echo(f"Modelo guardado em: {model_path}")

@app.command()
def tips(history: str = typer.Option(..., help="CSV de históricos"),
         fixtures: str = typer.Option(..., help="CSV de fixtures"),
         config: str = typer.Option("configs/default.yaml", help="Config YAML"),
         model_path: str = typer.Option("models/model.joblib", help="Modelo treinado (.joblib)"),
         out: str = typer.Option("outputs/tips.csv", help="Caminho CSV de saída")):
    cfg = load_config(config)
    hist = load_matches(history)
    fx = load_fixtures(fixtures)
    model = load_model(model_path)
    tips_df = generate_tips(hist, fx, model, cfg)
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tips_df.to_csv(out, index=False)
    typer.echo(f"Gerado {len(tips_df)} tips → {out}")
    if len(tips_df) > 0:
        typer.echo(tips_df.to_string(index=False))

@app.command()
def eval(history: str = typer.Option(..., help="CSV de históricos"),
         config: str = typer.Option("configs/default.yaml", help="Config YAML")):
    cfg = load_config(config)
    df = load_matches(history)
    feats = build_features(df, cfg.elo.start, cfg.elo.k_base, cfg.elo.surface_k_boost, cfg.features.form_window, cfg.features.h2h_decay)
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
    from sklearn.model_selection import train_test_split
    from .model import train_logreg_calibrated

    X = feats[["elo_diff","elo_surf_diff","form_diff","h2h_diff"]].values
    y = feats["y"].values

    model, metrics = train_logreg_calibrated(X, y, cfg.model.test_size, cfg.model.shuffle, cfg.model.max_iter, cfg.model.calibration, cfg.seed)
    # Simple report
    typer.echo(f"Eval — log_loss={metrics['log_loss']:.4f} | AUC={metrics['auc']:.4f} | n={metrics['samples_train']+metrics['samples_val']}")

if __name__ == "__main__":
    app()


@app.command()
def fetch(config: str = typer.Option("configs/data_sources.yaml", help="Config de fontes de dados"),
          sources: str = typer.Option("both", help="sackmann|tennisdata|both")):
    if sources in ("sackmann","both"):
        files = fetch_sackmann(config)
        typer.echo(f"Descargados {len(files)} ficheiros Sackmann para data/raw/")
    if sources in ("tennisdata","both"):
        files2 = fetch_tennis_data(config)
        typer.echo(f"Descargados {len(files2)} ficheiros Tennis-Data para data/raw/")
    typer.echo("✅ Fetch completo.")

@app.command()
def builddataset(config: str = typer.Option("configs/data_sources.yaml", help="Config de fontes de dados")):
    out = build_merged_dataset(config)
    typer.echo(f"✅ Dataset consolidado criado em: {out}")
