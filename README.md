
# 🎾 TennisTips — Professional Tennis Tipster

Modular Python package to train a probabilistic model (Elo + engineered features) and generate betting value tips.

> ⚠️ Educational purposes only. Bet responsibly and legally.

## Install (editable)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Quick start
Train on example data and produce tips:
```bash
tennistips train --history data/example/matches.csv --config configs/default.yaml --model-path models/model.joblib
tennistips tips --history data/example/matches.csv --fixtures data/example/fixtures.csv --config configs/default.yaml --model-path models/model.joblib --out outputs/tips.csv
```

## Layout
```
src/tennistips/        # package source
configs/               # YAML configs
data/example/          # sample CSVs
tests/                 # pytest
```

## Commands
- `tennistips train` — trains and saves a calibrated logistic model.
- `tennistips tips` — generates tips for upcoming fixtures; outputs CSV.
- `tennistips eval` — optional offline evaluation on a holdout split.

See `configs/default.yaml` for tweakable hyperparameters.


## Data pipeline
```bash
# 1) Buscar dados brutos (resultados + odds)
tennistips fetch --config configs/data_sources.yaml --sources both

# 2) Construir dataset consolidado p/ treino
tennistips builddataset --config configs/data_sources.yaml

# 3) Treinar usando o merged
tennistips train --history data/processed/matches.csv --config configs/default.yaml --model-path models/model.joblib
```
> Fontes: Jeff Sackmann (resultados) + Tennis-Data (odds). O merge usa fuzzy match por nomes/data.
