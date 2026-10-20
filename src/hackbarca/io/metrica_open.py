"""Metrica Sports sample-data (github.com/metrica-sports/sample-data): banco externo (ADR-0005).

Datos publicos, no del reto. Se guardan en external_data/metrica/ (ignorado por git).
"""

from __future__ import annotations

import os
import shutil
import urllib.request

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DEFAULT_CACHE = os.path.join(_REPO_ROOT, "external_data", "metrica")
_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"

FILES = {
    "1": ("Sample_Game_1/Sample_Game_1_RawEventsData.csv",
          "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
          "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"),
    "2": ("Sample_Game_2/Sample_Game_2_RawEventsData.csv",
          "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
          "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"),
    "3": ("Sample_Game_3/Sample_Game_3_events.json",
          "Sample_Game_3/Sample_Game_3_metadata.xml",
          "Sample_Game_3/Sample_Game_3_tracking.txt"),
}


def download_metrica(game: str, cache_dir: str = DEFAULT_CACHE) -> list[str]:
    paths = []
    for rel in FILES[game]:
        path = os.path.join(cache_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            partial = path + ".part"
            with urllib.request.urlopen(_RAW + rel) as response, open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle)
            if os.path.getsize(partial) == 0:
                raise IOError("descarga vacia: " + rel)
            os.replace(partial, path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for game in FILES:
        download_metrica(game)
        print("ok", game)
