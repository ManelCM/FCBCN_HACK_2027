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


def team_only_gt(gt: np.ndarray, ball_x: np.ndarray) -> np.ndarray:
    """E114 (I3): SkillCorner no tiene etiqueta de balon muerto (su `possession.group` nulo es
    "desconocido", no D; lo documentan katenaball y jekkiechan). Su D sale de `ball_state` de kloppy y su
    balon falta sobre todo fuera de plano. Solo se conservan H/A con el balon visto en el tracking limpio;
    el resto pasa a X (se descarta al entrenar)."""
    from hackbarca.io.external import GT_D, GT_X
    out = np.asarray(gt).copy()
    out[(out == GT_D) | ~np.isfinite(ball_x)] = GT_X
    return out


def _save(df, path):
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, columns=np.array(df.columns, dtype=str), values=df.to_numpy(np.float32))
    os.replace(tmp, path)


def skillcorner_training_games(version, params, limit: int | None = None, team_only: bool = False, extras=()):
    """[(ruta de features en cache, gt)] de los partidos de SkillCorner ya descargados.

    team_only: GT solo con H/A y el balon visto (E114); las features y su cache no cambian.
    extras: familias de entrenador (como `tune_l3 --extras`). Se anaden a la cache base, que no se recalcula,
    en un fichero aparte con su propia clave."""
    from hackbarca.io.skillcorner import available, load_skillcorner_bench
    os.makedirs(FEATURE_CACHE, exist_ok=True)
    out = []
    for k, match_id in enumerate(available()[:limit]):
        match, gt = load_skillcorner_bench(match_id)
        if team_only:
            gt = team_only_gt(gt, match.ball_x)
        name = "sc" + match_id
        path = feature_path(name, SEED0 + k, version, False, params)
        if not os.path.exists(path):
            _save(features_for(degrade(match, params, seed=SEED0 + k), None, version=version), path)
        if extras:
            full = feature_path(name, SEED0 + k, version, False, params, extras=tuple(extras))
            if not os.path.exists(full):
                import pandas as pd

                from hackbarca.features.coach import extra_features
                from hackbarca.pipeline.tune_l3 import read_features
                degraded = degrade(match, params, seed=SEED0 + k)   # la misma semilla: el mismo degradado
                _save(pd.concat([read_features(path), extra_features(degraded, tuple(extras))], axis=1), full)
            path = full
        out.append((path, gt))
    return out


def dead_carry_games(version, params, names, extras=()):
    """E123 (I7): por partido del banco, una copia degradada con el balón muerto llevado a sacar
    (`io/dead_carry.py`), **solo para entrenar**. [(ruta de features en cache, gt original)].

    Misma semilla que el banco (la posición del partido). La clave de la caché lleva el código de
    `dead_carry.py`, así que cambiarlo invalida solo estas copias."""
    import hashlib

    import pandas as pd

    from hackbarca.io.dead_carry import degrade_with_carry
    from hackbarca.io.external import load_bench_game
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "io", "dead_carry.py")
    with open(src, "rb") as handle:
        tag = hashlib.sha1(handle.read().replace(b"\r\n", b"\n")).hexdigest()[:8]
    os.makedirs(FEATURE_CACHE, exist_ok=True)
    out = []
    for i, game in enumerate(names):
        match, gt = load_bench_game(game)
        path = feature_path("%s_carry%s" % (game, tag), i, version, False, params, extras=tuple(extras))
        if not os.path.exists(path):
            degraded = degrade_with_carry(match, params, seed=i)
            df = features_for(degraded, None, version=version)
            if extras:
                from hackbarca.features.coach import extra_features
                df = pd.concat([df, extra_features(degraded, tuple(extras))], axis=1)
            _save(df, path)
        out.append((path, gt))
    return out
