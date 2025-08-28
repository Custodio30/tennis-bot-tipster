# scripts/fetch_fixtures_sofascore.py
import argparse
import datetime as dt
import os
import sys
import time
import csv
import json
from typing import List, Dict, Any, Optional

import requests


SOFASCORE_BASE = "https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{date}"

# Mapeia nível e piso (best effort). Se não souber, cai no default.
SURFACE_MAP = {
    # Grand Slams
    "Australian Open": "Hard",
    "Roland Garros": "Clay",
    "French Open": "Clay",
    "Wimbledon": "Grass",
    "US Open": "Hard",
    # Alguns Masters 1000
    "Indian Wells": "Hard",
    "Miami": "Hard",
    "Monte Carlo": "Clay",
    "Madrid": "Clay",
    "Rome": "Clay",
    "Canada": "Hard",
    "Cincinnati": "Hard",
    "Shanghai": "Hard",
    "Paris": "Hard",
    # Exemplos ATP 500/250 (ajuste à vontade)
    "Barcelona": "Clay",
    "Hamburg": "Clay",
    "Queens Club": "Grass",
    "Halle": "Grass",
    "Stuttgart": "Grass",
    "Eastbourne": "Grass",
    "Washington": "Hard",
    "Tokyo": "Hard",
    "Basel": "Hard",
}

def infer_level(ev: Dict[str, Any]) -> str:
    """
    Tenta inferir ATP/WTA/CH/ITF a partir de metadados do Sofascore.
    Se não der, usa 'ATP' como default.
    """
    try:
        uniq = ev.get("tournament", {}).get("uniqueTournament", {})
        cat = uniq.get("category", {})  # às vezes ajuda (e.g., ATP, WTA)
        name = uniq.get("name", "") or ev.get("tournament", {}).get("name", "")
        # Heurísticas simples:
        lname = (name or "").lower()
        cslug = (cat.get("slug", "") or "").lower()
        cname = (cat.get("name", "") or "").lower()
        if "wta" in (lname + cslug + cname):
            return "WTA"
        if "challenger" in lname or "ch." in lname or "ch-" in lname or "challenger" in cname:
            return "CH"
        if "itf" in lname or "itf" in cname:
            return "ITF"
        # Se nada bate, assume ATP
        return "ATP"
    except Exception:
        return "ATP"

def infer_surface(ev: Dict[str, Any]) -> str:
    """
    Determina superfície pelo nome do torneio (heurística). Default 'Hard'.
    """
    try:
        tname = ev.get("tournament", {}).get("name", "") or \
                ev.get("tournament", {}).get("uniqueTournament", {}).get("name", "")
        for key, surf in SURFACE_MAP.items():
            if key.lower() in tname.lower():
                return surf
        return "Hard"
    except Exception:
        return "Hard"

def call(url: str) -> Optional[Dict[str, Any]]:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (compatible; tennis-bot-tipster)"
            })
            if r.status_code == 200:
                return r.json()
            # 429 / 5xx: espera e tenta de novo
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            # Outros códigos: aborta
            print(f"[warn] GET {url} -> HTTP {r.status_code}", file=sys.stderr)
            return None
        except requests.RequestException as e:
            print(f"[warn] GET {url} failed ({e}), retrying...", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))
    return None

def fetch_day(d: dt.date) -> List[Dict[str, Any]]:
    url = SOFASCORE_BASE.format(date=d.strftime("%Y-%m-%d"))
    data = call(url)
    if not data or "events" not in data:
        return []
    rows = []
    for ev in data["events"]:
        # Estrutura típica: ev['homeTeam']['name'] / ev['awayTeam']['name'] para tênis (single)
        # fallback para outros campos se necessário
        p1 = ev.get("homeTeam", {}).get("name") or ev.get("homePlayer", {}).get("name")
        p2 = ev.get("awayTeam", {}).get("name") or ev.get("awayPlayer", {}).get("name")
        if not p1 or not p2:
            # às vezes eventos por equipes ou formatos diferentes — ignora
            continue
        tourn = ev.get("tournament", {}).get("name", "") or \
                ev.get("tournament", {}).get("uniqueTournament", {}).get("name", "")
        row = {
            "date": d.strftime("%Y-%m-%d"),
            "tournament": tourn,
            "level": infer_level(ev),
            "surface": infer_surface(ev),
            "player1": p1,
            "player2": p2,
            # odds placeholder (2.00 / 2.00). Trocar depois por fonte real de odds.
            "odds_p1": 2.00,
            "odds_p2": 2.00,
        }
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser(description="Fetch tennis fixtures (today+tomorrow) from Sofascore (unofficial).")
    ap.add_argument("--out", default="data/fixtures/latest.csv", help="Output CSV path")
    ap.add_argument("--days", type=int, default=2, help="How many days starting today (default=2 => today & tomorrow)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    start = dt.date.today()
    all_rows: List[Dict[str, Any]] = []
    for i in range(args.days):
        day = start + dt.timedelta(days=i)
        rows = fetch_day(day)
        all_rows.extend(rows)

    # dedup (player1, player2, date) – simples
    seen = set()
    uniq_rows = []
    for r in all_rows:
        key = (r["date"], r["player1"], r["player2"])
        if key in seen:
            continue
        seen.add(key)
        uniq_rows.append(r)

    # escreve CSV
    fieldnames = ["date", "tournament", "level", "surface", "player1", "player2", "odds_p1", "odds_p2"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(uniq_rows)

    print(f"✅ Fixtures salvos em: {args.out} ({len(uniq_rows)} partidas)")

if __name__ == "__main__":
    main()
