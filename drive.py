"""Leitura de pastas/arquivos públicos do Google Drive, com cache em disco."""
import json
import os
import re
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)


def _drive_id(value: str) -> str:
    """ID a partir de um link do Drive/Sheets (…/folders/<id>, …/d/<id>) ou do próprio ID."""
    m = re.search(r"/(?:folders|d)/([\w-]+)", value)
    return m.group(1) if m else value.strip()


def _env(name: str) -> list[str]:
    values = [v for v in os.environ.get(name, "").replace("\n", ",").split(",") if v.strip()]
    if not values:
        raise RuntimeError(f"Variável {name} não definida. Copie .env.example para .env e preencha os links.")
    return [_drive_id(v) for v in values]


def __getattr__(name: str):
    """Links/IDs do .env, lidos só quando o modo Drive os pede (o modo upload não precisa do .env)."""
    if name == "GT_ROOTS":  # pastas com subpastas images/ e labels/, uma por lote
        return _env("GT_FOLDERS")
    if name == "PRED_ROOT":
        return _env("PREDICTIONS_FOLDER")[0]
    if name == "METRICS_SHEET":
        return _env("METRICS_SHEET")[0]
    raise AttributeError(name)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60).read()


def list_folder(folder_id: str) -> list[tuple[str, str]]:
    """[(id, nome)] das entradas de uma pasta pública."""
    cached = CACHE / f"folder_{folder_id}.json"
    if cached.exists():
        return [tuple(x) for x in json.loads(cached.read_text())]
    html = _get(f"https://drive.google.com/embeddedfolderview?id={folder_id}").decode()
    items = [(m.group(1), m.group(2)) for m in
             re.finditer(r'id="entry-([\w-]+)".*?flip-entry-title">([^<]*)', html, re.S)]
    cached.write_text(json.dumps(items))
    return items


def download(file_id: str) -> bytes:
    path = CACHE / f"file_{file_id}"
    if not path.exists():
        path.write_bytes(_get(
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"))
    return path.read_bytes()


def sheet_csv(sheet_id: str) -> str:
    path = CACHE / f"sheet_{sheet_id}.csv"
    if not path.exists():
        path.write_bytes(_get(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"))
    return path.read_text(encoding="utf-8")


def clear_cache() -> None:
    for p in CACHE.iterdir():
        p.unlink()
