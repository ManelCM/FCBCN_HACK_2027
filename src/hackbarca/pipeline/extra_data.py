"""Partidos extra **solo para entrenar** (M3, E054): SkillCorner open data degradado como el banco.

Cada partido se degrada con los parametros del banco (v3.1) y una semilla propia (200 + k) y sus
features se guardan en la misma cache que `tune_l3` (`feature_path`, la clave incluye el codigo de las
features). El banco de 10 partidos sigue siendo el que evalua: los extra entran en todos los pliegues,
nunca se evaluan, asi que las notas siguen siendo comparables con E049.
"""

from __future__ import annotations

import os

import numpy as np

from hackbarca.io.degrade import degrade
from hackbarca.models.l3 import features_for
from hackbarca.pipeline.tune_l3 import FEATURE_CACHE, feature_path

SEED0 = 200


def skillcorner_training_games(version, params, limit: int | None = None):
    """[(ruta de features en cache, gt)] de los partidos de SkillCorner ya descargados."""
    from hackbarca.io.skillcorner import available, load_skillcorner_bench
    os.makedirs(FEATURE_CACHE, exist_ok=True)
    out = []
    for k, match_id in enumerate(available()[:limit]):
        match, gt = load_skillcorner_bench(match_id)
        name = "sc" + match_id
        path = feature_path(name, SEED0 + k, version, False, params)
        if not os.path.exists(path):
            df = features_for(degrade(match, params, seed=SEED0 + k), None, version=version)
            tmp = path[:-4] + ".tmp.npz"
            np.savez(tmp, columns=np.array(df.columns, dtype=str), values=df.to_numpy(np.float32))
            os.replace(tmp, path)
        out.append((path, gt))
    return out
