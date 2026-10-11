"""Shared helpers for downloading and parsing CUAD."""

import re
from pathlib import Path
from urllib.request import urlretrieve

CUAD_JSON_URL = (
    "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"
)


def normalize_question(question: str) -> str:
    match = re.search(r'related to "([^"]+)"', question)
    if match:
        return match.group(1)
    match = re.search(r'parties to the "([^"]+)"', question)
    if match:
        return match.group(1)
    return question


def download_cuad_json(raw_dir: str | Path = "data/raw/cuad") -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / "CUAD_v1.json"
    if json_path.exists():
        return json_path

    print("Downloading CUAD JSON from HuggingFace...")
    urlretrieve(CUAD_JSON_URL, json_path)
    return json_path
