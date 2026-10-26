"""U1 a fondo (E056): jugadores visibles en el broadcast como senal de balon muerto, solo agregados.

    scripts/run_aggregate.sh u1 uv run python -m hackbarca.pipeline.broadcast_shot_report

E055: la senal separa D de vivo con AUC 0,80 (prediccion segura) y 0,90 (ventana publica con GT).
Para simularla en el banco (patron de io/video_fill) hace falta su forma, sin GT del reto (con la
prediccion segura como pseudoetiqueta y la ventana publica como comprobacion):
  - distribucion del nº de jugadores visibles en el broadcast con el balon vivo y muerto;
  - duracion de los "planos" (rachas de >= K jugadores) y de los cortes;
  - retraso de la realizacion: cuanto tarda el broadcast en dejar el plano abierto tras un paso a D
    y en volver tras una reanudacion (agregado alrededor de las transiciones predichas seguras);
  - AUC dentro de los huecos del ATD y en los D cortos (que fraccion de los D cortos va con plano
    abierto, es decir, probablemente juego vivo).
Lee data/derived/real_signals.npz (lo escribe real_signals) y no imprime nada por frame.
"""

from __future__ import annotations

import json
import os

import numpy as np

from hackbarca.eval.evaluate import load_gt_codes, pm
from hackbarca.pipeline.atd_stats import GRADED, PUBLIC_GT
from hackbarca.pipeline.real_signals import _REPO_ROOT, _data, auc, contexts

H, A, D = 0, 1, 2
KEY = "U1 jugadores visibles en el broadcast (menos = paron)"
OUT = os.path.join(_REPO_ROOT, "docs", "investigacion", "resultados", "u1_broadcast.json")


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def main() -> int:
    from hackbarca.io.load_match import load_match
    match = load_match()
    count = -np.load(_data("derived", "real_signals.npz"))[KEY]
    pred = pm.label_codes(pm.read_label_csv(os.path.join(_REPO_ROOT, "submissions", "La MasIA.csv")))
    pmax = np.load(_data("derived", "proba_l3.npz"))["proba"].max(1)
    public = load_gt_codes(PUBLIC_GT)
    known = ~np.isnan(match.ball_x)
    ctx = contexts(pred, pmax, known, public)
    rep = {}
    for name, mask in (("vivo seguro", ctx["live_sure"]), ("D largo seguro", ctx["dead_sure"]),
                       ("D corto", ctx["short_d"]), ("publica vivo", ctx["pub_live"]), ("publica D", ctx["pub_dead"])):
        v = count[mask & np.isfinite(count)]
        rep[name] = dict(n=int(len(v)), p10=float(np.percentile(v, 10)), p50=float(np.percentile(v, 50)),
                         p90=float(np.percentile(v, 90)), share_ge10=float(np.mean(v >= 10)), share_le3=float(np.mean(v <= 3)))
        print("{:<16} n={:7d} jugadores p10/p50/p90 {:4.0f}/{:4.0f}/{:4.0f} | >=10 {:5.1%} | <=3 {:5.1%}".format(
            name, *[rep[name][k] for k in ("n", "p10", "p50", "p90", "share_ge10", "share_le3")]))
    gap = ctx["gap"]
    rep["auc_hueco_seguro"] = auc(-count[ctx["dead_sure"] & gap], -count[ctx["live_sure"] & gap])
    rep["auc_hueco_publica"] = auc(-count[ctx["pub_dead"] & gap], -count[ctx["pub_live"] & gap])
    print("AUC dentro de los huecos del ATD: seguro %.3f | publica %.3f" % (rep["auc_hueco_seguro"], rep["auc_hueco_publica"]))
    graded = ctx["graded"] & np.isfinite(count)
    for k in (8, 10, 14):
        wide = graded & (count >= k)
        lw, lc = runs(wide), runs(graded & (count < k))
        rep["planos_%d" % k] = dict(abiertos_p50_s=float(np.median(lw) / 25), cortes_p50_s=float(np.median(lc) / 25),
                                    abiertos_por_min=float(len(lw) / (graded.sum() / 1500)))
        print("umbral %2d: planos abiertos p50 %.1f s, cortes p50 %.1f s, %.2f planos abiertos por minuto" % (
            k, *rep["planos_%d" % k].values()))
    # D cortos: fraccion de tramos cuyo plano es abierto (mediana >= 10) -> probablemente vivos
    short_runs = []
    for a, b in GRADED:
        for s, e, c in pm.segments(pred, [(a, b)]):
            if c == D and e - s < 125 and np.isfinite(count[s:e]).any():
                short_runs.append(float(np.nanmedian(count[s:e])))
    short_runs = np.array(short_runs)
    rep["D_cortos_con_plano_abierto"] = float(np.mean(short_runs >= 10)) if len(short_runs) else float("nan")
    print("D cortos predichos: %d tramos, con plano abierto (mediana >= 10 jugadores) %.1f%%" % (
        len(short_runs), 100 * rep["D_cortos_con_plano_abierto"]))
    # retraso de la realizacion alrededor de transiciones seguras (paso a D y reanudacion)
    for label, frm, to in (("paso a D", (H, A), D), ("reanudacion", (D,), None)):
        prof = []
        for a, b in GRADED:
            w = pred[a:b]
            for t in np.flatnonzero(w[1:] != w[:-1]) + 1:
                t0 = a + t
                if w[t - 1] in frm and (to is None and w[t] in (H, A) or w[t] == to) and pmax[t0 - 50:t0 + 50].min() >= 0.9:
                    if t0 + 250 < len(count) and t0 - 250 > 0:
                        prof.append(count[t0 - 250:t0 + 251])
        prof = np.array(prof)
        if len(prof):
            med = np.nanmedian(prof, axis=0)
            rep["perfil_" + label] = med[::25].tolist()
            print("%s (%d transiciones seguras): jugadores medianos cada 1 s de -10 a +10 s: %s" % (
                label, len(prof), " ".join("%.0f" % v for v in med[::25])))
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(rep, handle, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
