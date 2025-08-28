
import os, io, time
import requests
from tqdm import tqdm
import yaml

def _download(url: str, dest_path: str, session: requests.Session, retries:int=3):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(retries):
        r = session.get(url, timeout=60)
        if r.status_code == 200 and len(r.content) > 0:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return True
        time.sleep(1 + attempt)
    return False

def fetch_sackmann(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    base = cfg["sackmann"]["base_url"]
    pattern = cfg["sackmann"]["pattern"]
    wta_base = cfg["sackmann"]["wta_base_url"]
    wta_pattern = cfg["sackmann"]["wta_pattern"]
    years = cfg["sackmann"]["years"]
    outdir = cfg["paths"]["raw_dir"]
    s = requests.Session()

    downloaded = []
    for y in tqdm(years, desc="Sackmann ATP"):
        url = f"{base}/{pattern.format(year=y)}"
        dest = os.path.join(outdir, f"atp_matches_{y}.csv")
        if _download(url, dest, s):
            downloaded.append(dest)
    for y in tqdm(years, desc="Sackmann WTA"):
        url = f"{wta_base}/{wta_pattern.format(year=y)}"
        dest = os.path.join(outdir, f"wta_matches_{y}.csv")
        _download(url, dest, s)  # optional; ignore failure for older years

    return downloaded

def fetch_tennis_data(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    base = cfg["tennis_data"]["base_url"]
    atp_p = cfg["tennis_data"]["atp_pattern"]
    wta_p = cfg["tennis_data"]["wta_pattern"]
    years = cfg["tennis_data"]["years"]
    outdir = cfg["paths"]["raw_dir"]
    s = requests.Session()
    downloaded = []
    for y in tqdm(years, desc="Tennis-Data ATP"):
        url = f"{base}{atp_p.format(year=y)}"
        dest = os.path.join(outdir, f"tennisdata_atp_{y}.csv")
        if _download(url, dest, s):
            downloaded.append(dest)
    for y in tqdm(years, desc="Tennis-Data WTA"):
        url = f"{base}{wta_p.format(year=y)}"
        dest = os.path.join(outdir, f"tennisdata_wta_{y}.csv")
        _download(url, dest, s)
    return downloaded
