
.PHONY: install train tips test

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e .

train:
	tennistips train --history data/example/matches.csv --config configs/default.yaml --model-path models/model.joblib

tips:
	tennistips tips --history data/example/matches.csv --fixtures data/example/fixtures.csv --config configs/default.yaml --model-path models/model.joblib --out outputs/tips.csv

test:
	pytest -q
