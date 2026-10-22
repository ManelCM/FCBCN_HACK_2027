"""Datos abiertos DFL/IDSSE (Bassek et al., CC-BY 4.0): banco de validacion externo (ADR-0005).

Son datos publicos, no del reto: se pueden inspeccionar libremente. Se guardan en
external_data/dfl/<match_id>/ (ignorado por git), fuera de data/.
"""

from __future__ import annotations

import os
import shutil
import urllib.request

from kloppy._providers.sportec import get_IDSSE_url

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DEFAULT_CACHE = os.path.join(_REPO_ROOT, "external_data", "dfl")

DFL_MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
_KINDS = ("meta", "event", "tracking")


def dfl_paths(match_id: str, cache_dir: str = DEFAULT_CACHE) -> dict[str, str]:
    folder = os.path.join(cache_dir, match_id)
    return {kind: os.path.join(folder, "{}.xml".format(kind)) for kind in _KINDS}


def download_dfl(match_id: str, cache_dir: str = DEFAULT_CACHE) -> dict[str, str]:
    """Descarga (si falta) meta, eventos y tracking de un partido y devuelve sus rutas."""
    paths = dfl_paths(match_id, cache_dir)
    os.makedirs(os.path.dirname(paths["meta"]), exist_ok=True)
    for kind, path in paths.items():
        if os.path.exists(path):
            continue
        partial = path + ".part"
        with urllib.request.urlopen(get_IDSSE_url(match_id, kind)) as response, \
                open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle)
        # figshare responde 202 con cuerpo vacio a clientes no-navegador (reto anti-bots)
        if os.path.getsize(partial) == 0:
            os.remove(partial)
            raise IOError("figshare devolvio un fichero vacio para {} {}: descargalo con un "
                          "navegador y dejalo en {}".format(match_id, kind, path))
        os.replace(partial, path)
    return paths


# Nombre con que figshare sirve cada fichero, p. ej.
# DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml -> meta
_NAME_KEYS = {"matchinformation": "meta", "events": "event", "positions": "tracking"}


def collect_downloads(src_dir: str, cache_dir: str = DEFAULT_CACHE) -> list[tuple[str, str]]:
    """Mueve los XML de DFL bajados con un navegador a external_data/dfl/<id>/<tipo>.xml."""
    moved = []
    for name in sorted(os.listdir(src_dir)):
        if not (name.startswith("DFL_") and name.endswith(".xml")):
            continue
        match = next((m for m in DFL_MATCHES if name.endswith("DFL-MAT-{}.xml".format(m))), None)
        kind = next((k for key, k in _NAME_KEYS.items() if key in name), None)
        if match is None or kind is None:
            continue
        target = dfl_paths(match, cache_dir)[kind]
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(os.path.join(src_dir, name), target)
        moved.append((match, kind))
    return moved


def missing_files(cache_dir: str = DEFAULT_CACHE) -> list[tuple[str, str]]:
    return [(m, k) for m in DFL_MATCHES for k, p in dfl_paths(m, cache_dir).items()
            if not os.path.exists(p)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-downloads", default=None,
                        help="carpeta donde el navegador dejo los XML (figshare bloquea urllib)")
    args = parser.parse_args()
    if args.from_downloads:
        for match, kind in collect_downloads(args.from_downloads):
            print("movido", match, kind)
        print("faltan:", missing_files() or "nada")
    else:
        for match in DFL_MATCHES:
            download_dfl(match)
            print("ok", match)
