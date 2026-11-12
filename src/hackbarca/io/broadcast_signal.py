"""U1 (E056-E057): jugadores visibles en el broadcast como feature, real en el reto y simulada en el banco.

La realizacion de television deja el plano abierto mientras se juega y corta a primeros planos y
repeticiones en los parones. En el reto (E056, solo agregados):
  - vivo: >= 10 jugadores visibles el 90-96 % del tiempo, <= 3 el 4-9 %;
  - muerto: >= 10 el 24-29 %, <= 3 el 69-76 %;
  - planos abiertos (>= 10) de mediana ~13 s y cortes de ~6 s;
  - dentro de los huecos del ATD separa D de vivo con AUC 0,78 (prediccion segura) y 0,91 (ventana
    publica con GT).
Los partidos del banco no tienen broadcast: se simula con la GT del propio banco y esa forma (patron
de io/video_fill). Proceso de dos estados (plano abierto / corte) con rachas lognormales cuya
probabilidad de abrir depende de si el balon esta vivo, un retraso de realizacion de 0-3 s en las
transiciones de la GT y el numero de jugadores visibles sacado de distribuciones por estado.
(io/broadcast_sim.py es otra cosa: el broadcast sintetico para probar la alineacion.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FPS = 25
GT_D, GT_X = 2, 3


@dataclass(frozen=True)
class BroadcastSignal:
    wide_live: float = 0.93        # cuota de plano abierto con el balon vivo
    wide_dead: float = 0.27        # ... con el balon muerto
    wide_run_s: float = 15.0       # mediana de un plano abierto (s)
    run_sigma: float = 0.8         # dispersion lognormal de las rachas
    lag_max_s: float = 3.0         # retraso de la realizacion tras una transicion vivo/muerto
    wide_mean: float = 17.0        # jugadores visibles en plano abierto (media, sd)
    wide_sd: float = 3.5
    cut_mean: float = 0.8          # ... en un corte (media de una geometrica desplazada)
    cut_min_s: float = 3.0         # con el balon muerto, un corte (primer plano, repeticion) dura al menos esto


BROADCAST_SIGNAL = BroadcastSignal()


def simulate_broadcast_count(gt: np.ndarray, sim: BroadcastSignal = BROADCAST_SIGNAL, seed: int = 0) -> np.ndarray:
    """Jugadores visibles en el broadcast por frame (NaN donde la GT es X)."""
    rng = np.random.default_rng(seed + 7919)
    n = len(gt)
    dead = gt == GT_D
    state = dead.copy()          # estado que ve la realizacion: con retraso en cada transicion
    for t in np.flatnonzero(np.diff(dead.astype(np.int8)) != 0) + 1:
        state[t:t + int(rng.uniform(0, sim.lag_max_s) * FPS)] = dead[t - 1]
    count = np.zeros(n)
    changes = np.concatenate((np.flatnonzero(np.diff(state.astype(np.int8)) != 0) + 1, [n]))
    t = 0
    is_wide = True
    while t < n:
        p_wide = sim.wide_dead if state[t] else sim.wide_live
        # planos abiertos de mediana fija; la del corte se ajusta para que la cuota sea p_wide
        cut_min = sim.cut_min_s if state[t] else 1.0      # en juego los cortes son breves
        med = sim.wide_run_s if is_wide else float(np.clip(sim.wide_run_s * (1 - p_wide) / max(p_wide, 1e-3), cut_min, 60))
        length = max(1, int(np.exp(rng.normal(np.log(med * FPS), sim.run_sigma))))
        # la realizacion corta en los cambios vivo/muerto: el plano acaba como mucho ahi
        boundary = int(changes[np.searchsorted(changes, t, side="right")])
        stop = min(t + length, boundary)
        # el nivel se sortea por plano (en el reto la cuenta es estable dentro de un plano) y solo
        # varia +-1 jugador de un frame a otro
        if is_wide:
            level = float(np.clip(rng.normal(sim.wide_mean, sim.wide_sd), 11, 22))
            count[t:stop] = np.clip(np.round(level + rng.integers(-1, 2, stop - t)), 10, 22)
        else:
            level = float(rng.geometric(1.0 / (sim.cut_mean + 1.0)) - 1)
            count[t:stop] = np.clip(np.round(level + rng.integers(-1, 2, stop - t)), 0, 8)
        if stop == boundary and stop < n:
            # nuevo estado: abre o corta con la cuota del estado que empieza
            is_wide = bool(rng.random() < (sim.wide_dead if state[stop] else sim.wide_live))
        else:
            is_wide = not is_wide
        t = stop
    count[gt == GT_X] = np.nan
    return count


def _centered_mean(x: np.ndarray, half: int) -> np.ndarray:
    ok = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(x))
    a, b = np.clip(idx - half, 0, len(x)), np.clip(idx + half + 1, 0, len(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cn[b] - cn[a] > 0, (cs[b] - cs[a]) / (cn[b] - cn[a]), np.nan)


def broadcast_features(count: np.ndarray) -> pd.DataFrame:
    """Columnas de la familia "broadcast" a partir de los jugadores visibles por frame."""
    wide = np.where(np.isfinite(count), (count >= 10).astype(float), np.nan)
    f = {"bc_n": count}
    for half in (12, 75, 250):
        f["bc_n_m%d" % half] = _centered_mean(count, half)
        f["bc_wide_m%d" % half] = _centered_mean(wide, half)
    m = _centered_mean(wide, 37)
    before = np.full(len(count), np.nan)
    after = np.full(len(count), np.nan)
    before[75:] = m[:-75]
    after[:-75] = m[75:]
    f["bc_wide_chg75"] = after - before
    return pd.DataFrame(f).astype(np.float32)


def shape(count: np.ndarray, gt: np.ndarray) -> dict:
    """Los agregados de E056, para calibrar la simulacion contra el reto."""
    out = {}
    for name, mask in (("vivo", (gt == 0) | (gt == 1)), ("D", gt == GT_D)):
        v = count[mask & np.isfinite(count)]
        out[name] = dict(ge10=round(float(np.mean(v >= 10)), 3), le3=round(float(np.mean(v <= 3)), 3))
    graded = np.isfinite(count)
    wide = graded & (count >= 10)

    def runs(mask):
        e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
        return np.flatnonzero(e == -1) - np.flatnonzero(e == 1)

    out["abiertos_p50_s"] = round(float(np.median(runs(wide)) / FPS), 1)
    out["cortes_p50_s"] = round(float(np.median(runs(graded & ~wide)) / FPS), 1)
    return out


def shift_report() -> int:
    """Desajuste reto/banco de las columnas bc_* (solo percentiles; ADR-0004)."""
    import os

    from hackbarca.io.external import load_bench_game
    from hackbarca.pipeline.atd_stats import GRADED
    from hackbarca.pipeline.bench_config import BENCH_GAMES_V3
    from hackbarca.pipeline.feature_shift import shift_table
    key = "U1 jugadores visibles en el broadcast (menos = paron)"
    count = -np.load(os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), "derived", "real_signals.npz"))[key]
    graded = np.zeros(len(count), dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    real = broadcast_features(count)[graded]
    bench = pd.concat([broadcast_features(simulate_broadcast_count(load_bench_game(n)[1], seed=i))
                       for i, n in enumerate(BENCH_GAMES_V3)], ignore_index=True)
    for col, pr, pb, gap in sorted(shift_table(real, bench, list(real.columns)), key=lambda r: -r[3]):
        print("{:<16} reto {:>18} banco {:>18} desaj. {:.2f}".format(
            col, "/".join("%.2f" % v for v in pr), "/".join("%.2f" % v for v in pb), gap))
    return 0


def main() -> int:
    """Calibracion: forma de la simulacion en el banco frente a la del reto (E056)."""
    import sys
    if "--shift" in sys.argv:
        return shift_report()
    from hackbarca.io.external import load_bench_game
    from hackbarca.pipeline.bench_config import BENCH_GAMES_V3
    target = {"vivo": dict(ge10=0.90, le3=0.09), "D": dict(ge10=0.29, le3=0.69),
              "abiertos_p50_s": 13.3, "cortes_p50_s": 5.8}
    counts, gts = [], []
    for i, name in enumerate(BENCH_GAMES_V3):
        _, raw = load_bench_game(name)
        counts.append(np.concatenate([simulate_broadcast_count(raw, seed=i), [np.nan]]))
        gts.append(np.concatenate([raw, [GT_X]]))
    print("objetivo (reto):", target)
    print("banco simulado: ", shape(np.concatenate(counts), np.concatenate(gts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
