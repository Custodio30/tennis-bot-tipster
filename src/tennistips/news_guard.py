# src/tennistips/news_guard.py
from __future__ import annotations
import re, time
from typing import Dict, List, Set, Tuple
import feedparser
import requests
from bs4 import BeautifulSoup

KEYWORDS = [
    r"withdraws?", r"withdrawal", r"pulls out", r"retires?", r"retirement",
    r"walkover", r"injur(?:y|ed)", r"ruled out", r"out of", r"abandona", r"les(ã|a)o"
]
KW_RE = re.compile("|".join(KEYWORDS), flags=re.IGNORECASE)

DEFAULT_FEEDS = [
    # ATP / WTA oficiais
    "https://www.atptour.com/en/media/rss-feed",            # ATP News RSS
    "https://www.wtatennis.com/news/1350431/rss-feeds",     # WTA Feeds hub (vários links na página)
    # Media gerais (podem ter lag, mas cobrem Majors)
    "https://www.espn.com/espn/rss/tennis/news",            # ESPN tennis RSS
    "https://www.skysports.com/rss/12024",                  # Sky Sports Tennis RSS
]

# Tens fontes HTML sem RSS (ex.: Tennis Explorer injured) -> scraper leve:
HTML_SOURCES = [
    ("tennisexplorer_injured", "https://www.tennisexplorer.com/list-players/injured/"),
]

def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[.\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_player_names(text: str) -> List[str]:
    # heurística rápida: “Firstname Lastname” com inicial maiúscula
    # (podes trocar por dicionário de nomes dos teus CSVs)
    cand = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text or "")
    # filtra palavras comuns
    bad = {"US Open","ATP Tour","WTA Tour","Sky Sports","ESPN"}
    return [c for c in cand if c not in bad]

def fetch_rss_urls_from_hub(url: str) -> List[str]:
    """Algumas páginas (ex. WTA hub) listam várias RSS; extraímos os links."""
    try:
        html = requests.get(url, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        feeds = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "rss" in href and href.startswith("http"):
                feeds.append(href)
        return list(dict.fromkeys(feeds))
    except Exception:
        return []

def parse_feed(url: str) -> List[Tuple[str,str]]:
    """Devolve lista de (title, link)."""
    try:
        d = feedparser.parse(url)
        items = []
        for e in d.entries:
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            items.append((title, link))
        return items
    except Exception:
        return []

def scrape_injured_page(url: str) -> List[str]:
    """Extrai nomes de jogadores listados como injured/retired (Tennis Explorer)."""
    try:
        html = requests.get(url, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        names = []
        for a in soup.select("table a"):
            t = a.get_text(" ", strip=True)
            if t and len(t.split()) >= 2:
                names.append(t)
        return names
    except Exception:
        return []

def build_news_flags(player_index: Set[str], extra_feeds: List[str] = None) -> Dict[str, Dict]:
    """
    player_index: nomes normalizados que existem no teu histórico/fixtures (para reduzir falsos positivos).
    Return: {"player_norm": {"hits":[(title,link)], "reason":"kw|scrape"}}
    """
    feeds = []
    for f in DEFAULT_FEEDS:
        if f.endswith("rss-feeds"):  # WTA hub
            feeds.extend(fetch_rss_urls_from_hub(f))
        else:
            feeds.append(f)
    if extra_feeds:
        feeds.extend(extra_feeds)

    flagged: Dict[str, Dict] = {}

    # 1) RSS news scanning
    for f in feeds:
        for title, link in parse_feed(f):
            if not (title and KW_RE.search(title)):
                continue
            for name in extract_player_names(title):
                n = _norm_name(name)
                if n in player_index:
                    flagged.setdefault(n, {"hits": [], "reason": "news"} )
                    flagged[n]["hits"].append((title, link))

    # 2) Injured/retired list scraping
    for key, url in HTML_SOURCES:
        for name in scrape_injured_page(url):
            n = _norm_name(name)
            if n in player_index:
                flagged.setdefault(n, {"hits": [], "reason": "injury_list"})
                flagged[n]["hits"].append((f"{name} listed injured/retired", url))

    return flagged
