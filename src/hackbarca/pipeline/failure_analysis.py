"""Analisis a fondo de los fallos del modelo vigente (L3 apilado, E018) en el banco v2.1.

    uv run python -m hackbarca.pipeline.failure_analysis [--oof oof_l3_notags_f3_stack.npz]

Solo datos externos (Metrica). Usa las probabilidades fuera de muestra que guarda `tune_l3` y la
decodificacion elegida en E019 (DecodeParams por defecto). Responde a:
1. que posesiones se pierden y por que (taxonomia de cada posesion GT no emparejada),
2. en que contexto (duracion, huecos de balon, como empiezan y acaban, zona del campo),
3. como de bien se colocan las fronteras segun el tipo de transicion,
4. donde falla el frame a frame y por que (senal del mas cercano, fase de juego, calibracion),
5. cuanto spellF1 valdria arreglar cada causa (oraculos).
Guarda el resumen en external_data/failure_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict

import numpy as np

from hackbarca.eval.bench import pm
from hackbarca.io.degrade import degrade
from hackbarca.io.external import CONTROL_TYPES, controlled_possession, load_metrica_bench, metrica_events
from hackbarca.models.l1 import interpolate_ball, team_distances
from hackbarca.models.l3 import DecodeParams, decode
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
H, A, D, X = 0, 1, 2, 3
LONG_GAP = 100   # frames (4 s): hueco "largo", como en atd_stats
SHORT_SPELL = 75  # 3 s
IOU = 0.5        # umbral titular de spellF1
DURATIONS = ((0, 75, "<3 s"), (75, 250, "3-10 s"), (250, 750, "10-30 s"), (750, 10 ** 9, ">30 s"))

CATEGORIES = (
    "equipo contrario",
    "tapada por D",
    "fusionada: paron no detectado",
    "fusionada: posesion rival corta no detectada",
    "fusionada: ambos",
    "partida en trozos",
    "fronteras desplazadas o recortadas",
    "consumida por otra",
)


# ----------------------------------------------------------------------------- utilidades

def concat(arrays):
    sep = np.array([X], dtype=np.int8)
    return np.concatenate([np.concatenate([a.astype(np.int8), sep]) for a in arrays])


def spell(gts, preds) -> dict:
    g, p = concat(gts), concat(preds)
    return pm.spell_scores(g, p, pm.valid_runs(g))


def runs_of(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def greedy_pairs(overlaps, threshold):
    """{gt_index: pred_index} del emparejamiento voraz oficial a un umbral."""
    pairs, used = {}, set()
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if i in pairs or j in used:
            continue
        pairs[i] = j
        used.add(j)
    return pairs


def _restart_name(subtype: str) -> str:
    s = subtype.upper()
    for key, name in (("THROW", "saque de banda"), ("CORNER", "corner"), ("KICK OFF", "saque de centro"),
                      ("GOAL KICK", "saque de puerta"), ("PENALTY", "penalti"), ("FREE", "falta")):
        if key in s:
            return name
    return "otra reanudacion"


def start_kind(events, s: int, team: int, prev_label: int) -> str:
    """Como empieza una posesion GT: reanudacion (tras D) o tipo de robo (tras el rival)."""
    near = events[(events["start"] >= s - 2) & (events["start"] <= s + 2) & (events["team"] == team)
                  & events["type"].isin(CONTROL_TYPES)]
    if not len(near):
        return "sin evento"
    ev = near.iloc[0]
    if ev["type"] == "SET PIECE":
        return "reanudacion: " + _restart_name(ev["subtype"])
    if ev["type"] == "PASS" and "GOAL KICK" in ev["subtype"]:
        return "reanudacion: saque de puerta"
    if prev_label == D:
        return "tras D sin saque marcado"
    if ev["type"] == "RECOVERY":
        sub = ev["subtype"]
        if "INTERCEPTION" in sub:
            return "robo: intercepcion"
        if "THEFT" in sub:
            return "robo: entrada"
        if "SAVED" in sub or "BLOCKED" in sub:
            return "robo: parada o bloqueo"
        return "robo: balon suelto"
    before = events[(events["start"] < s) & (events["start"] >= s - 50)]
    if len(before) and (before["type"] == "CHALLENGE").any():
        return "robo: tras duelo"
    return "robo: otro (" + ev["type"].lower() + ")"


def end_kind(events, s: int, e: int, team: int, next_label: int) -> str:
    """Como acaba una posesion GT: en paron (por que) o en perdida (de que tipo)."""
    inside = events[(events["start"] >= s) & (events["start"] < e + 2)]
    own = inside[inside["team"] == team]
    if next_label == D:
        if (inside["type"] == "BALL OUT").any():
            return "paron: balon fuera"
        if (inside["type"] == "SHOT").any():
            return "paron: tiro"
        if (inside["type"] == "FAULT RECEIVED").any() or inside["subtype"].str.contains("FAULT").any():
            return "paron: falta"
        return "paron: otro"
    if next_label not in (H, A):
        return "fin de parte"
    lost = own[own["type"] == "BALL LOST"]
    if len(lost):
        sub = lost.iloc[-1]["subtype"]
        if "INTERCEPTION" in sub:
            return "perdida: pase interceptado"
        if "THEFT" in sub:
            return "perdida: robo"
        if "HEAD" in sub:
            return "perdida: cabezazo"
        if "CLEARANCE" in sub:
            return "perdida: despeje"
        return "perdida: otra"
    if (own["type"] == "SHOT").any():
        return "perdida: tiro sin paron"
    if (inside["type"] == "CHALLENGE").any():
        return "perdida: duelo"
    return "perdida: sin evento"


# ----------------------------------------------------------------------------- datos

def load(oof_name: str, params: DecodeParams, controlled: float = 0.0):
    blob = np.load(os.path.join(_REPO_ROOT, "external_data", oof_name))
    games = []
    for i, name in enumerate(BENCH_GAMES):
        match, gt = load_metrica_bench(name)
        assert np.array_equal(gt, blob["gt%d" % i]), "los OOF no corresponden a esta GT"
        if controlled:
            gt = controlled_possession(gt, int(round(controlled * match.fps)))
        deg = degrade(match, BENCH_DEGRADE, seed=i)
        proba = blob["p%d" % i]
        pred = decode(proba, pm.valid_runs(gt), params)

        bx, by, _ = interpolate_ball(deg, 10)
        known = ~np.isnan(bx)
        long_gap = np.zeros(match.n_frames, dtype=bool)
        for a, b in runs_of(np.isnan(deg.ball_x)):
            if b - a > LONG_GAP:
                long_gap[a:b] = True
        dh, da = team_distances(deg, np.nan_to_num(bx), np.nan_to_num(by))
        near = np.where(dh <= da, H, A)
        near[~known | ~(np.isfinite(dh) | np.isfinite(da))] = X
        cbx, cby, _ = interpolate_ball(match, 10)
        ch, ca = team_distances(match, np.nan_to_num(cbx), np.nan_to_num(cby))
        near_clean = np.where(ch <= ca, H, A)
        near_clean[np.isnan(cbx)] = X

        events = metrica_events(name)
        in_pass = np.zeros(match.n_frames, dtype=bool)
        for _, ev in events[events["type"] == "PASS"].iterrows():
            in_pass[int(ev["start"]) + 1:int(ev["end"])] = True
        near_duel = np.zeros(match.n_frames, dtype=bool)
        for _, ev in events[events["type"] == "CHALLENGE"].iterrows():
            near_duel[max(0, int(ev["start"]) - 12):int(ev["start"]) + 13] = True

        games.append(dict(gt=gt, proba=proba, pred=pred, known=known, long_gap=long_gap, near=near,
                          near_clean=near_clean, in_pass=in_pass, near_duel=near_duel, events=events,
                          ball_x=match.ball_x, ball_y=match.ball_y, length=match.pitch_length,
                          width=match.pitch_width))
    return games


# ----------------------------------------------------------------------------- taxonomia

def classify_spells(game) -> list[dict]:
    gt, pred, events = game["gt"], game["pred"], game["events"]
    runs = pm.valid_runs(gt)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gs, ps)
    pairs = greedy_pairs(overlaps, IOU)
    best = defaultdict(list)
    for iou, i, j in overlaps:
        best[i].append((iou, j))
    segs = pm.segments(gt, runs)
    seg_index = {(a, b): n for n, (a, b, _) in enumerate(segs)}

    out = []
    for i, (s, e, k) in enumerate(gs):
        n = seg_index[(s, e)]
        prev_label = segs[n - 1][2] if n > 0 and segs[n - 1][1] == s else X
        next_label = segs[n + 1][2] if n + 1 < len(segs) and segs[n + 1][0] == e else X
        other = A if k == H else H
        seg = pred[s:e]
        f_own, f_other, f_dead = (np.mean(seg == k), np.mean(seg == other), np.mean(seg == D))
        bx = game["ball_x"][s:e]
        x0 = bx[~np.isnan(bx)][0] if (~np.isnan(bx)).any() else np.nan
        row = dict(start=s, stop=e, dur=e - s, team=k, matched=i in pairs,
                   gap_frac=float(game["long_gap"][s:e].mean()),
                   start_kind=start_kind(events, s, k, prev_label),
                   end_kind=end_kind(events, s, e, k, next_label),
                   passes=int(((events["type"] == "PASS") & (events["team"] == k) & (events["start"] >= s)
                               & (events["start"] < e)).sum()),
                   x0=float(x0) / game["length"] if np.isfinite(x0) else np.nan,
                   category="ok")
        if not row["matched"]:
            cands = sorted(best[i], reverse=True)
            if cands and cands[0][0] >= IOU:
                row["category"] = "consumida por otra"
            elif f_other >= 0.5:
                row["category"] = "equipo contrario"
            elif f_dead >= 0.5:
                row["category"] = "tapada por D"
            else:
                j = cands[0][1] if cands else None
                ps0, pe0 = (ps[j][0], ps[j][1]) if j is not None else (s, e)
                covers = (min(e, pe0) - max(s, ps0)) / (e - s)
                if j is not None and pe0 - ps0 > e - s and covers >= 0.7:
                    span = gt[ps0:pe0]
                    outside = np.concatenate([span[:max(0, s - ps0)], span[max(0, e - ps0):]])
                    has_d, has_other = (outside == D).any(), (outside == other).any()
                    row["category"] = ("fusionada: ambos" if has_d and has_other else
                                       "fusionada: posesion rival corta no detectada" if has_other else
                                       "fusionada: paron no detectado")
                    row["merge_span"] = (int(ps0), int(pe0))
                elif sum(1 for _, jj in cands if ps[jj][2] == k) >= 2:
                    row["category"] = "partida en trozos"
                else:
                    row["category"] = "fronteras desplazadas o recortadas"
        out.append(row)
    return out


def false_spells(game) -> Counter:
    """Posesiones predichas sin pareja a IoU 0,5, por tipo."""
    gt, pred = game["gt"], game["pred"]
    runs = pm.valid_runs(gt)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gs, ps)
    used = set(greedy_pairs(overlaps, IOU).values())
    touched = Counter(j for _, _, j in overlaps)
    out = Counter()
    for j, (s, e, k) in enumerate(ps):
        if j in used:
            continue
        if not touched[j]:
            kind = "equipo inexistente (sin GT del mismo equipo debajo)"
        elif e - s < SHORT_SPELL:
            kind = "trozo corto (<3 s) de una posesion real"
        else:
            kind = "solapa una real pero mal ajustada"
        out[kind] += 1
    return out


# ----------------------------------------------------------------------------- fronteras

def transitions(codes, runs):
    out = []
    for s, e in runs:
        w = codes[s:e]
        idx = np.flatnonzero(w[1:] != w[:-1]) + 1
        out += [(s + int(t), int(w[t - 1]), int(w[t])) for t in idx]
    return out


def ttype(a, b) -> str:
    if {a, b} == {H, A}:
        return "cambio de equipo (H<->A)"
    if a == D:
        return "reanudacion (D->H/A)"
    return "paron (H/A->D)"


def boundary_stats(games) -> dict:
    res = defaultdict(lambda: dict(n=0, hit12=0, hit25=0, hit50=0, offsets=[]))
    false = Counter()
    for g in games:
        runs = pm.valid_runs(g["gt"])
        gt_t, pr_t = transitions(g["gt"], runs), transitions(g["pred"], runs)
        by_type = defaultdict(list)
        for t, a, b in pr_t:
            by_type[(a, b)].append(t)
        by_type = {key: np.array(v) for key, v in by_type.items()}
        gt_by_type = defaultdict(list)
        for t, a, b in gt_t:
            gt_by_type[(a, b)].append(t)
            r = res[ttype(a, b)]
            r["n"] += 1
            cand = by_type.get((a, b))
            if cand is None or not len(cand):
                continue
            off = cand[np.argmin(np.abs(cand - t))] - t
            r["hit12"] += abs(off) <= 12
            r["hit25"] += abs(off) <= 25
            r["hit50"] += abs(off) <= 50
            if abs(off) <= 50:
                r["offsets"].append(int(off))
        for t, a, b in pr_t:
            ref = np.array(gt_by_type.get((a, b), []))
            if not len(ref) or np.min(np.abs(ref - t)) > 50:
                false[ttype(a, b)] += 1
    out = {}
    for key, r in res.items():
        o = np.array(r["offsets"]) if r["offsets"] else np.zeros(1)
        out[key] = dict(n=r["n"], within_0_5s=r["hit12"] / r["n"], within_1s=r["hit25"] / r["n"],
                        within_2s=r["hit50"] / r["n"], median_offset=float(np.median(o)),
                        mae_frames=float(np.mean(np.abs(o))), false_within_2s=false[key])
    return out


# ----------------------------------------------------------------------------- oraculos

def oracle(games, masks) -> float:
    preds = [np.where(m, g["gt"], g["pred"]).astype(np.int8) for g, m in zip(games, masks)]
    return spell([g["gt"] for g in games], preds)["mean_f1"]


def category_masks(games, spells, cats):
    masks = []
    for g, rows in zip(games, spells):
        m = np.zeros(len(g["gt"]), dtype=bool)
        for r in rows:
            if r["category"] in cats:
                a, b = r.get("merge_span", (r["start"], r["stop"]))
                m[min(a, r["start"]):max(b, r["stop"])] = True
        masks.append(m)
    return masks


# ----------------------------------------------------------------------------- main

LONG_PASS = 30.0   # m
SWITCH_DY = 25.0   # m de desplazamiento lateral para llamarlo cambio de juego
AFTER = 25         # frames tras la recepcion o la perdida en que se mide si ya acertamos el equipo


def pass_kind(ev) -> str | None:
    """Tipo de envio de un evento PASS (completado) o BALL LOST (perdido), por longitud."""
    if ev["type"] not in ("PASS", "BALL LOST") or not np.isfinite([ev["sx"], ev["ex"]]).all():
        return None
    outcome = "completado" if ev["type"] == "PASS" else "perdido"
    if ev["type"] == "BALL LOST" and "CLEARANCE" in str(ev["subtype"]):
        return "despeje"
    length = np.hypot(ev["ex"] - ev["sx"], ev["ey"] - ev["sy"])
    if length >= LONG_PASS:
        kind = "cambio de juego" if abs(ev["ey"] - ev["sy"]) >= SWITCH_DY else "balon largo"
    else:
        kind = "medio (15-30 m)" if length >= 15 else "corto (< 15 m)"
    return "{} {}".format(kind, outcome)


def pass_length_report(games) -> dict:
    """Acierto durante el vuelo y en el segundo siguiente, por tipo de envio (cambios de juego,
    balones largos, despejes). Solo frames con la GT viva."""
    stats = {}
    for g in games:
        gt, pred = g["gt"], g["pred"]
        alive = (gt == H) | (gt == A)
        for _, ev in g["events"].iterrows():
            kind = pass_kind(ev)
            start, end = int(ev["start"]), int(ev["end"])
            if kind is None or end <= start:
                continue
            s = stats.setdefault(kind, dict(n=0, fly_ok=0, fly=0, after_ok=0, after=0))
            s["n"] += 1
            fly = slice(start + 1, end)
            after = slice(end, end + AFTER)
            s["fly_ok"] += int(np.sum((pred[fly] == gt[fly]) & alive[fly]))
            s["fly"] += int(np.sum(alive[fly]))
            s["after_ok"] += int(np.sum((pred[after] == gt[after]) & alive[after]))
            s["after"] += int(np.sum(alive[after]))
    out = {}
    for kind in sorted(stats, key=lambda k: -stats[k]["n"]):
        s = stats[kind]
        fly = s["fly_ok"] / max(s["fly"], 1)
        after = s["after_ok"] / max(s["after"], 1)
        print("  {:<34} n={:5d}  acierto en vuelo {:.3f}  en el segundo siguiente {:.3f}".format(
            kind, s["n"], fly, after))
        out[kind] = dict(n=s["n"], in_flight=fly, next_second=after)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default="oof_l3_notags_f3_stack.npz")
    parser.add_argument("--controlled", type=float, default=0.0,
                        help="segundos minimos de control para que un robo cuente (E020); 0 = GT por eventos")
    args = parser.parse_args(argv)
    params = DecodeParams()
    games = load(args.oof, params, args.controlled)
    gts = [g["gt"] for g in games]
    report = {"oof": args.oof, "decode": asdict(params), "controlled_s": args.controlled}

    # 1. nota
    base = spell(gts, [g["pred"] for g in games])
    print("== 1. Nota ({}; decodificacion {}; GT {})".format(
        args.oof, params, "controlada %g s" % args.controlled if args.controlled else "por eventos"))
    print("spellF1 {:.4f}  posesiones predichas {} / GT {}".format(base["mean_f1"], base["n_pred"], base["n_gt"]))
    for t in (0.3, 0.5, 0.7):
        r = base["per_iou"][t]
        print("  IoU {:.1f}: precision {:.3f}  recall {:.3f}  F1 {:.3f}".format(t, r["precision"], r["recall"], r["f1"]))
    report["spell"] = base["mean_f1"]
    report["n_pred"], report["n_gt"] = base["n_pred"], base["n_gt"]
    report["per_iou"] = {str(k): v for k, v in base["per_iou"].items()}

    # 2. taxonomia
    spells = [classify_spells(g) for g in games]
    rows = [r for s in spells for r in s]
    n = len(rows)
    missed = [r for r in rows if not r["matched"]]
    print("\n== 2. Posesiones GT perdidas a IoU 0,5: {} de {} ({:.1%})".format(len(missed), n, len(missed) / n))
    cat_count = Counter(r["category"] for r in missed)
    cat_frames = Counter()
    for r in missed:
        cat_frames[r["category"]] += r["dur"]
    report["taxonomy"] = {}
    for c in CATEGORIES:
        print("  {:<46} {:4d} ({:5.1%} de las perdidas)  duracion mediana {:5.1f} s".format(
            c, cat_count[c], cat_count[c] / max(len(missed), 1),
            np.median([r["dur"] for r in missed if r["category"] == c] or [0]) / 25))
        report["taxonomy"][c] = cat_count[c]
    fs = Counter()
    for g in games:
        fs.update(false_spells(g))
    print("  posesiones predichas sobrantes (falsos positivos):")
    for k, v in fs.most_common():
        print("    {:<52} {:4d}".format(k, v))
    report["false_spells"] = dict(fs)

    # 3. contexto
    def table(title, key_fn, order=None):
        groups = defaultdict(lambda: [0, 0])
        for r in rows:
            k = key_fn(r)
            groups[k][0] += int(r["matched"])
            groups[k][1] += 1
        keys = order or sorted(groups, key=lambda k: -groups[k][1])
        print("\n  " + title)
        out = {}
        for k in keys:
            ok, tot = groups[k]
            if not tot:
                continue
            print("    {:<40} {:4d} posesiones  encontradas {:5.1%}  perdidas {:4d}".format(str(k), tot, ok / tot, tot - ok))
            out[str(k)] = dict(n=tot, found=ok / tot)
        return out

    print("\n== 3. Contexto de las posesiones GT (encontradas a IoU 0,5)")

    def dur_key(r):
        return next(lab for lo, hi, lab in DURATIONS if lo <= r["dur"] < hi)

    report["by_duration"] = table("por duracion", dur_key, [d[2] for d in DURATIONS])
    report["by_gap"] = table("por % de la posesion en hueco largo de balon",
                             lambda r: "0 %" if r["gap_frac"] == 0 else "1-50 %" if r["gap_frac"] <= 0.5 else ">50 %",
                             ["0 %", "1-50 %", ">50 %"])
    report["by_start"] = table("por como empieza", lambda r: r["start_kind"])
    report["by_end"] = table("por como acaba", lambda r: r["end_kind"])
    report["by_passes"] = table("por numero de pases propios",
                                lambda r: "0" if r["passes"] == 0 else "1-2" if r["passes"] <= 2 else "3-5" if r["passes"] <= 5 else "6+",
                                ["0", "1-2", "3-5", "6+"])
    report["by_zone"] = table("por zona donde empieza (x relativa del balon limpio)",
                              lambda r: "sin balon" if not np.isfinite(r["x0"]) else
                              "tercio 1" if r["x0"] < 1 / 3 else "tercio 2" if r["x0"] < 2 / 3 else "tercio 3",
                              ["tercio 1", "tercio 2", "tercio 3", "sin balon"])
    cross = Counter((dur_key(r), r["category"]) for r in missed)
    print("\n  perdidas por duracion x causa:")
    for lo, hi, lab in DURATIONS:
        parts = ["{} {}".format(c.split(":")[0][:14], cross[(lab, c)]) for c in CATEGORIES if cross[(lab, c)]]
        print("    {:<8} ".format(lab) + " | ".join(parts))

    # 4. fronteras
    print("\n== 4. Fronteras (transiciones GT con una predicha del mismo tipo cerca)")
    bstats = boundary_stats(games)
    for k, v in bstats.items():
        print("  {:<26} n={:4d}  a <=0,5 s {:5.1%}  <=1 s {:5.1%}  <=2 s {:5.1%}  desfase mediano {:+.0f} fr  "
              "error medio {:.1f} fr  falsas {:4d}".format(k, v["n"], v["within_0_5s"], v["within_1s"],
                                                            v["within_2s"], v["median_offset"], v["mae_frames"],
                                                            v["false_within_2s"]))
    report["boundaries"] = bstats

    # 5. frame a frame
    print("\n== 5. Frame a frame")
    contexts = (("balon visible", lambda g: g["known"]),
                ("hueco corto (<=4 s)", lambda g: ~g["known"] & ~g["long_gap"]),
                ("hueco largo (>4 s)", lambda g: g["long_gap"]))
    report["frames"] = {}
    for name, fn in contexts:
        cm = np.zeros((3, 3))
        for g in games:
            m = fn(g) & (g["gt"] != X)
            for a in range(3):
                for b in range(3):
                    cm[a, b] += np.sum((g["gt"][m] == a) & (g["pred"][m] == b))
        tot = cm.sum()
        rows_n = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        print("  {:<20} {:5.1%} de los frames  acierto {:.3f}".format(name, tot / sum((g["gt"] != X).sum() for g in games),
                                                                      np.trace(cm) / tot))
        print("     GT H -> pred H/A/D {:.2f} {:.2f} {:.2f} | GT A -> {:.2f} {:.2f} {:.2f} | GT D -> {:.2f} {:.2f} {:.2f}".format(
            *rows_n[0], *rows_n[1], *rows_n[2]))
        report["frames"][name] = dict(share=tot, acc=float(np.trace(cm) / tot), confusion=rows_n.tolist())

    alive = [(g["gt"] == H) | (g["gt"] == A) for g in games]
    for label, fn in (("en vuelo (dentro de un pase)", lambda g: g["in_pass"]),
                      ("cerca de un duelo (+-0,5 s)", lambda g: g["near_duel"] & ~g["in_pass"]),
                      ("resto del juego vivo", lambda g: ~g["near_duel"] & ~g["in_pass"])):
        hit = tot = 0
        for g, al in zip(games, alive):
            m = fn(g) & al
            hit += np.sum(g["pred"][m] == g["gt"][m])
            tot += m.sum()
        print("  fase: {:<32} frames {:7d}  acierto {:.3f}".format(label, tot, hit / max(tot, 1)))

    print("\n  senal 'jugador mas cercano = poseedor' (juego vivo con balon visible):")
    report["nearest"] = {}
    for label, key in (("balon degradado", "near"), ("balon limpio", "near_clean")):
        right = tot = hit_r = hit_w = n_w = 0
        for g, al in zip(games, alive):
            m = al & (g[key] != X)
            ok = g[key][m] == g["gt"][m]
            right += ok.sum()
            tot += m.sum()
            pm_ok = g["pred"][m] == g["gt"][m]
            hit_r += np.sum(pm_ok & ok)
            hit_w += np.sum(pm_ok & ~ok)
            n_w += np.sum(~ok)
        print("    {:<16} acierta {:.3f}  | modelo cuando el mas cercano acierta {:.3f}, cuando falla {:.3f}".format(
            label, right / tot, hit_r / max(right, 1), hit_w / max(n_w, 1)))
        report["nearest"][label] = dict(acc=right / tot, model_when_right=hit_r / max(right, 1),
                                        model_when_wrong=hit_w / max(n_w, 1))

    print("\n  calibracion (probabilidad maxima de la capa final vs acierto del argmax):")
    report["calibration"] = {}
    for lo, hi in ((0.33, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        hit = tot = 0
        for g in games:
            m = g["gt"] != X
            pmax = g["proba"][m].max(1)
            sel = (pmax >= lo) & (pmax < hi)
            hit += np.sum(np.argmax(g["proba"][m][sel], 1) == g["gt"][m][sel])
            tot += sel.sum()
        print("    p en [{:.2f}, {:.2f}): {:5.1%} de los frames  acierto {:.3f}".format(
            lo, min(hi, 1), tot / sum((g["gt"] != X).sum() for g in games), hit / max(tot, 1)))
        report["calibration"]["%.2f-%.2f" % (lo, min(hi, 1))] = hit / max(tot, 1)

    # 6. oraculos
    print("\n== 6. Oraculos: spellF1 si arreglaramos solo esto (base {:.4f})".format(base["mean_f1"]))
    rng = np.random.default_rng(0)
    half_gaps = []
    for g in games:
        m = np.zeros(len(g["gt"]), dtype=bool)
        for a, b in runs_of(g["long_gap"]):
            if rng.random() < 0.5:
                m[a:b] = True
        half_gaps.append(m)
    short = []
    for g, rows_g in zip(games, spells):
        m = np.zeros(len(g["gt"]), dtype=bool)
        for r in rows_g:
            if r["dur"] < SHORT_SPELL:
                m[r["start"]:r["stop"]] = True
        short.append(m)
    scenarios = [
        ("huecos largos perfectos (balon fusionado ideal)", [g["long_gap"] for g in games]),
        ("la mitad de los huecos largos perfectos", half_gaps),
        ("huecos cortos perfectos", [~g["known"] & ~g["long_gap"] for g in games]),
        ("D perfecto (frames D)", [g["gt"] == D for g in games]),
        ("equipo perfecto (frames vivos)", alive),
        ("posesiones cortas (<3 s) perfectas", short),
        ("frames en vuelo perfectos", [g["in_pass"] for g in games]),
    ]
    scenarios += [("causa: " + c, category_masks(games, spells, {c})) for c in CATEGORIES if cat_count[c]]
    report["oracles"] = {}
    for name, masks in scenarios:
        v = oracle(games, masks)
        print("  {:<58} {:.4f}  ({:+.4f})".format(name, v, v - base["mean_f1"]))
        report["oracles"][name] = v

    # 7. pases por longitud
    print("\n== 7. Pases por longitud (acierto frame a frame con el juego vivo)")
    report["passes"] = pass_length_report(games)

    suffix = "_controlled%g" % args.controlled if args.controlled else ""
    path = os.path.join(_REPO_ROOT, "external_data", "failure_analysis%s.json" % suffix)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False, default=float)
    print("\nresumen en", os.path.relpath(path, _REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
