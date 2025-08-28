# scripts/fetch_fixtures.py
import argparse
import csv
import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Protocol

import requests

def iso_dates_from_today(days: int) -> List[str]:
    today = dt.date.today()
    return [(today + dt.timedelta(days=i)).isoformat() for i in range(days)]

def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def write_csv(path: str, rows: List[Dict[str, Any]]):
    ensure_dir_for(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "event_id", "tournament", "category", "surface", "home", "away", "start_ts"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] Gravado: {path} ({len(rows)} linhas)")

def norm_row(date_iso, event_id, tournament, category, surface, home, away, start_ts):
    return {
        "date": date_iso,
        "event_id": event_id or "",
        "tournament": tournament or "",
        "category": category or "",
        "surface": surface or "",
        "home": home or "",
        "away": away or "",
        "start_ts": start_ts,
    }

class Provider(Protocol):
    def fetch_day(self, date_iso: str) -> List[Dict[str, Any]]:
        ...

# ---------------------------
# Provider 1: SportsDataIO
# ---------------------------
class SportsDataIOProvider:
    BASE = "https://api.sportsdata.io/v4/tennis/scores/json"

    def __init__(self):
        key = os.getenv("SPORTSDATAIO_KEY")
        if not key:
            raise RuntimeError("Define a env var SPORTSDATAIO_KEY com a tua chave do SportsDataIO.")
        self.headers = {"Ocp-Apim-Subscription-Key": key}

    @staticmethod
    def _to_sdi_date(date_iso: str) -> str:
        # SportsDataIO usa formato: 2025-AUG-29
        d = dt.date.fromisoformat(date_iso)
        return d.strftime("%Y-%b-%d").upper()

    def fetch_day(self, date_iso: str) -> List[Dict[str, Any]]:
        url = f"{self.BASE}/GamesByDate/{self._to_sdi_date(date_iso)}"
        r = requests.get(url, headers=self.headers, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"SportsDataIO HTTP {r.status_code}: {r.text[:200]}")
        data = r.json() or []
        out = []
        for g in data:
            # Campos mais comuns; em trial podem vir “scrambled”
            event_id = g.get("GameId")
            tournament = g.get("Tournament") or g.get("League", "")
            category = g.get("Country") or ""
            home = g.get("HomeTeam") or ""
            away = g.get("AwayTeam") or ""
            day_iso = g.get("Day")  # ISO8601
            start_ts = None
            if isinstance(day_iso, str):
                try:
                    start_ts = int(dt.datetime.fromisoformat(day_iso.replace("Z", "+00:00")).timestamp())
                except Exception:
                    start_ts = None
            surface = g.get("Surface") or ""
            out.append(norm_row(date_iso, event_id, tournament, category, surface, home, away, start_ts))
        return out

# --------------------------------------------
# Provider 2: SofaScore (HTML público, sem API)
# --------------------------------------------
class SofaScoreHTMLProvider:
    # Página de calendário diário, ex.: https://www.sofascore.com/tennis/2025-08-29
    BASE = "https://www.sofascore.com/tennis/{date}"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,pt;q=0.8",
        "Referer": "https://www.sofascore.com/",
    }

    def _extract_json(self, html: str) -> Optional[Dict[str, Any]]:
        """
        Muitos sites Next.js/NUXT injetam um JSON grande em <script> (ex.: __NEXT_DATA__, __NUXT__).
        Tentamos achar um blob com 'events' para parse.
        """
        # 1) Procurar __NEXT_DATA__ ou __NUXT__:
        for pat in [r'id="__NEXT_DATA__"\s*type="application/json">(.+?)</script>',
                    r'window\.__NUXT__\s*=\s*(\{.+?\});']:
            m = re.search(pat, html, re.DOTALL)
            if m:
                raw = m.group(1)
                try:
                    return json.loads(raw)
                except Exception:
                    try:
                        # às vezes termina com ;, retira
                        return json.loads(raw.rstrip(";"))
                    except Exception:
                        pass
        # 2) fallback “bruto”: procurar um array "events":[{...}]
        m = re.search(r'"events"\s*:\s*\[(.*?)\]\s*[,}]', html, re.DOTALL)
        if m:
            txt = "[" + m.group(1) + "]"
            try:
                return {"events": json.loads(txt)}
            except Exception:
                pass
        return None

    def _harvest_events_from_blob(self, blob: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Tentar caminhos comuns até obter uma lista de eventos
        candidates = []
        if "events" in blob and isinstance(blob["events"], list):
            candidates = blob["events"]
        else:
            # Percorrer profundamente procurando uma key "events"
            def dfs(obj):
                out = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "events" and isinstance(v, list):
                            out.extend(v)
                        else:
                            out.extend(dfs(v))
                elif isinstance(obj, list):
                    for it in obj:
                        out.extend(dfs(it))
                return out
            candidates = dfs(blob)
        events = []
        for ev in candidates:
            if not isinstance(ev, dict):
                continue
            # heurística mínima: exige id + equipas/jogadores
            event_id = ev.get("id") or ev.get("event", {}).get("id")
            if not event_id:
                continue
            tour = (ev.get("tournament") or {}).get("name") or ""
            cat = (ev.get("tournament") or {}).get("category", {}).get("name") or ""
            home = (ev.get("homeTeam") or ev.get("homePlayer") or {}).get("name") or ""
            away = (ev.get("awayTeam") or ev.get("awayPlayer") or {}).get("name") or ""
            start_ts = ev.get("startTimestamp") or (ev.get("event") or {}).get("startTimestamp")
            surface = ""
            events.append((event_id, tour, cat, surface, home, away, start_ts))
        # dedupe por event_id
        seen = set()
        uniq = []
        for row in events:
            if row[0] in seen:
                continue
            seen.add(row[0])
            uniq.append(row)
        return uniq

    def fetch_day(self, date_iso: str) -> List[Dict[str, Any]]:
        url = self.BASE.format(date=date_iso)
        r = requests.get(url, headers=self.HEADERS, timeout=25)
        if r.status_code != 200:
            raise RuntimeError(f"SofaScore HTML HTTP {r.status_code}")
        blob = self._extract_json(r.text)
        if not blob:
            raise RuntimeError("Não consegui extrair JSON da página pública do SofaScore.")
        rows = self._harvest_events_from_blob(blob)
        return [
            norm_row(date_iso, ev_id, tour, cat, surf, home, away, start_ts)
            for (ev_id, tour, cat, surf, home, away, start_ts) in rows
        ]

# ------------------------------------------------------
# Provider 3: SofaScore via Playwright (headless browser)
# ------------------------------------------------------
class SofaScorePlaywrightProvider:
    BASE = "https://www.sofascore.com/tennis/{date}"

    def __init__(self):
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Playwright não está instalado. Faz: pip install playwright && playwright install"
            ) from e
        self.sync_playwright = __import__("playwright.sync_api", fromlist=["sync_playwright"]).sync_playwright

    def fetch_day(self, date_iso: str) -> List[Dict[str, Any]]:
        with self.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(locale="en-US")
            page = context.new_page()
            page.goto(self.BASE.format(date=date_iso), wait_until="networkidle", timeout=60000)

            # Tenta ler um dos blobs globais
            js = """
            () => {
              try { if (window.__NEXT_DATA__) return window.__NEXT_DATA__; } catch(e){}
              try { if (window.__NUXT__) return window.__NUXT__; } catch(e){}
              return null;
            }
            """
            blob = page.evaluate(js)
            browser.close()
        if not blob:
            raise RuntimeError("Não encontrei dados no contexto da página (Playwright).")
        # Reaproveita a extração do provider HTML para uniformizar
        helper = SofaScoreHTMLProvider()
        rows = helper._harvest_events_from_blob(blob)
        return [
            norm_row(date_iso, ev_id, tour, cat, surf, home, away, start_ts)
            for (ev_id, tour, cat, surf, home, away, start_ts) in rows
        ]

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=1, help="Quantos dias a partir de hoje (inclui hoje). Use 2 para amanhã.")
    ap.add_argument("--provider", choices=["sportsdataio", "sofascore_html", "sofascore_playwright"],
                    default="sofascore_html")
    args = ap.parse_args()

    if args.provider == "sportsdataio":
        provider: Provider = SportsDataIOProvider()
    elif args.provider == "sofascore_playwright":
        provider = SofaScorePlaywrightProvider()
    else:
        provider = SofaScoreHTMLProvider()

    dates = iso_dates_from_today(args.days)
    rows: List[Dict[str, Any]] = []
    for d in dates:
        print(f"[info] A buscar fixtures de {d} via {args.provider}...")
        rows.extend(provider.fetch_day(d))
        time.sleep(0.7)  # civilidade

    write_csv(args.out, rows)

if __name__ == "__main__":
    main()
