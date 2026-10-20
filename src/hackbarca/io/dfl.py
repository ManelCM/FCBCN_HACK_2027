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


if __name__ == "__main__":
    for match in DFL_MATCHES:
        download_dfl(match)
        print("ok", match)
