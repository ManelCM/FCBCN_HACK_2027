"""Banco de validacion externo (ADR-0005): partidos publicos -> MatchTracking + GT H/A/D.

Metrica sample-data no trae los flags de "balon muerto" y "equipo duenio" del tracking corregido
(de los que sale la GT del reto), asi que la GT se deriva de sus eventos con `gt_from_events`.
La regla esta documentada en docs/decisiones/0006-banco-externo.md.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from hackbarca.io.load_match import AWAY, HOME, MatchTracking
from hackbarca.io.metrica_open import DEFAULT_CACHE as METRICA_CACHE
from hackbarca.io.metrica_open import FILES as METRICA_FILES
from hackbarca.io.metrica_open import download_metrica

# Codigos de possession_metrics: indice en LABELS = ("H", "A", "D", "X")
GT_H, GT_A, GT_D, GT_X = 0, 1, 2, 3
PITCH = (105.0, 68.0)

# Eventos en los que el equipo que lo ejecuta tiene el balon controlado. CHALLENGE no cuenta:
# un duelo no es control (CHALLENGE.md: "an aerial duel won without subsequent control").
CONTROL_TYPES = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "CARRY", "BALL LOST"}


def from_kloppy(dataset, pitch=PITCH) -> MatchTracking:
    """TrackingDataset (coordenadas 0-1) -> MatchTracking en metros, indexado por frame_id - 1."""
    length, width = pitch
    df = dataset.to_df(engine="pandas")
    n_frames = int(df["frame_id"].max())
    index = df["frame_id"].to_numpy() - 1

    players, teams = [], []
    for team in dataset.metadata.teams:
        side = HOME if team is dataset.metadata.teams[0] else AWAY
        for player in team.players:
            if "{}_x".format(player.player_id) in df.columns:
                players.append(player.player_id)
                teams.append(side)

    def full(values, scale):
        out = np.full((n_frames,) + values.shape[1:], np.nan)
        out[index] = values * scale
        return out

    px = full(df[["{}_x".format(p) for p in players]].to_numpy(float), length)
    py = full(df[["{}_y".format(p) for p in players]].to_numpy(float), width)
    period = np.zeros(n_frames, dtype=np.int8)
    period[index] = df["period_id"].to_numpy()

    return MatchTracking(
        n_frames=n_frames, fps=int(round(dataset.metadata.frame_rate)),
        pitch_length=length, pitch_width=width,
        team_of_track=np.array(teams, dtype=np.int8),
        players_x=px, players_y=py, players_speed=np.full_like(px, np.nan),
        ball_x=full(df["ball_x"].to_numpy(float), length),
        ball_y=full(df["ball_y"].to_numpy(float), width),
        ball_speed=np.full(n_frames, np.nan),
        period=period,
    )


def _coord(point, axis: str) -> float:
    value = (point or {}).get(axis)
    return float(value) if value is not None else np.nan


def metrica_events(game: str, cache_dir: str = METRICA_CACHE) -> pd.DataFrame:
    """Eventos normalizados: team (HOME/AWAY), type, subtype, period, start, end (0-indexados) y
    posicion de inicio y fin en metros (sx, sy, ex, ey; NaN si el evento no la trae)."""
    path = os.path.join(cache_dir, METRICA_FILES[game][0])
    if path.endswith(".csv"):
        raw = pd.read_csv(path)
        events = pd.DataFrame({
            "team": np.where(raw["Team"] == "Home", HOME, AWAY),
            "type": raw["Type"], "subtype": raw["Subtype"].fillna(""),
            "period": raw["Period"],
            "start": raw["Start Frame"] - 1, "end": raw["End Frame"] - 1,
            "sx": raw["Start X"] * PITCH[0], "sy": raw["Start Y"] * PITCH[1],
            "ex": raw["End X"] * PITCH[0], "ey": raw["End Y"] * PITCH[1],
        })
    else:
        with open(path) as handle:
            data = json.load(handle)["data"]
        home_id = "FIFATMA"  # primer <Team> del metadata.xml del Game 3 = local

        def subtype(event):
            sub = event.get("subtypes")
            if isinstance(sub, list):
                return "-".join(s["name"] for s in sub)
            return sub["name"] if sub else ""

        events = pd.DataFrame({
            "team": [HOME if e["team"]["id"] == home_id else AWAY for e in data],
            "type": [e["type"]["name"] for e in data],
            "subtype": [subtype(e) for e in data],
            "period": [e["period"] for e in data],
            "start": [e["start"]["frame"] - 1 for e in data],
            "end": [(e["end"]["frame"] or e["start"]["frame"]) - 1 for e in data],
            "sx": [_coord(e["start"], "x") * PITCH[0] for e in data],
            "sy": [_coord(e["start"], "y") * PITCH[1] for e in data],
            "ex": [_coord(e["end"], "x") * PITCH[0] for e in data],
            "ey": [_coord(e["end"], "y") * PITCH[1] for e in data],
        })
    # Algunos eventos instantaneos traen End Frame = 0 en el CSV
    events["end"] = np.maximum(events["end"], events["start"])
    return events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)


def _dead_start(previous: pd.DataFrame) -> int | None:
    """Frame en que muere el balon antes de una reanudacion, mirando los eventos previos."""
    for _, event in previous[::-1].iterrows():
        if event["type"] == "CARD":
            continue
        if event["type"] == "FAULT RECEIVED" or "FAULT" in str(event["subtype"]):
            return int(event["start"])
        return int(event["end"])
    return None


def gt_from_events(events: pd.DataFrame, n_frames: int) -> np.ndarray:
    """GT H/A/D/X por frame a partir de eventos.

    - X fuera de [primer evento, ultimo evento] de cada periodo (sin GT, como el reto).
    - D desde que muere el balon (fin del evento previo, o la falta) hasta el frame anterior a
      cada SET PIECE.
    - H/A: equipo del ultimo evento de control (CONTROL_TYPES) iniciado en o antes del frame.
    """
    codes = np.full(n_frames, GT_X, dtype=np.int8)
    for _, half in events.groupby("period"):
        lo, hi = int(half["start"].min()), int(half["end"].max())
        control = half[half["type"].isin(CONTROL_TYPES)]
        owner = np.full(hi - lo + 1, -1, dtype=np.int8)
        starts = control["start"].to_numpy() - lo
        teams = control["team"].to_numpy()
        for start, team in zip(starts, teams):
            owner[start:] = team
        span = np.where(owner == HOME, GT_H, np.where(owner == AWAY, GT_A, GT_D))

        for row, event in half.iterrows():
            if event["type"] != "SET PIECE":
                continue
            previous = half.loc[:row - 1]
            previous = previous[previous["start"] < event["start"]]
            dead = _dead_start(previous)
            if dead is not None and dead < event["start"]:
                span[dead - lo:event["start"] - lo] = GT_D
        codes[lo:hi + 1] = span
    return codes


def controlled_possession(gt: np.ndarray, min_frames: int) -> np.ndarray:
    """GT con la regla de posesion "controlada" de CHALLENGE.md.

    Un robo solo cambia la posesion si el nuevo equipo la conserva al menos `min_frames`; si no,
    esos frames siguen siendo del equipo anterior (un toque, un desvio o un duelo sin control no
    cuentan). Tras un D la posesion vale siempre: al reanudar hay un poseedor claro. Los eventos de
    Metrica marcan cada recuperacion, asi que sin esto el banco tiene el doble de posesiones <3 s
    que la GT publica (E020).
    """
    out = gt.copy()
    edges = np.flatnonzero(np.diff(gt) != 0) + 1
    bounds = np.concatenate(([0], edges, [len(gt)]))
    prev = None  # etiqueta vigente de la posesion abierta
    for a, b in zip(bounds[:-1], bounds[1:]):
        code = gt[a]
        if code in (GT_H, GT_A) and prev in (GT_H, GT_A) and code != prev and b - a < min_frames:
            out[a:b] = prev  # robo sin control: sigue la posesion anterior
            continue
        prev = code
    return out


def load_metrica_bench(game: str, cache_dir: str = METRICA_CACHE):
    """(MatchTracking, gt_codes) de un partido de Metrica sample-data, con cache .npz."""
    cache = os.path.join(cache_dir, "bench_game{}.npz".format(game))
    if not os.path.exists(cache):
        from kloppy import metrica

        paths = download_metrica(game, cache_dir)
        if game == "3":
            dataset = metrica.load_tracking_epts(meta_data=paths[1], raw_data=paths[2])
        else:
            dataset = metrica.load_tracking_csv(home_data=paths[1], away_data=paths[2])
        match = from_kloppy(dataset)
        gt = gt_from_events(metrica_events(game, cache_dir), match.n_frames)
        np.savez_compressed(cache, gt=gt, **{k: v for k, v in vars(match).items()})
    return _load_cache(cache)


def _load_cache(cache: str):
    blob = np.load(cache)
    fields = {k: blob[k] for k in blob.files if k != "gt"}
    for scalar in ("n_frames", "fps"):
        fields[scalar] = int(fields[scalar])
    for scalar in ("pitch_length", "pitch_width"):
        fields[scalar] = float(fields[scalar])
    return MatchTracking(**fields), blob["gt"]


def gt_from_tracking(states, owners, home) -> np.ndarray:
    """GT H/A/D por frame a partir de los flags del tracking corregido, como la del reto.

    CHALLENGE.md: "a frame is D where the corrected tracking marks the ball dead, and otherwise
    takes the team the tracking names as owning it". `states` son "alive"/"dead" y `owners` el
    equipo duenio de cada frame (el objeto Team de kloppy o None).
    """
    dead = np.array([str(s).lower().endswith("dead") for s in states])
    is_home = np.array([o is home for o in owners])
    return np.where(dead, GT_D, np.where(is_home, GT_H, GT_A)).astype(np.int8)


def dfl_to_bench(dataset, pitch=PITCH):
    """(MatchTracking, gt) de un TrackingDataset de DFL cargado con coordinates="metrica".

    Los frame_id de DFL saltan entre partes (10000.., 100000..), asi que las filas se compactan y
    entre partes se deja un frame X para que possession_metrics las trate por separado. El balon
    se borra en los frames muertos para imitar el tracking limpio de Metrica, en el que falta en
    el 91-99 % de los D: asi `degrade` (calibrado sobre Metrica) se aplica igual.
    """
    records = dataset.records
    period = np.array([r.period.id for r in records])
    breaks = np.concatenate(([0], np.cumsum(np.diff(period) != 0)))
    index = np.arange(len(records)) + breaks  # un hueco de 1 frame en cada cambio de parte
    n = int(index[-1]) + 1
    home = dataset.metadata.teams[0]
    gt = np.full(n, GT_X, dtype=np.int8)
    gt[index] = gt_from_tracking([r.ball_state for r in records],
                                 [r.ball_owning_team for r in records], home)

    df = dataset.to_df(engine="pandas")
    players, teams = [], []
    for team in dataset.metadata.teams:
        for player in team.players:
            if "{}_x".format(player.player_id) in df.columns:
                players.append(player.player_id)
                teams.append(HOME if team is home else AWAY)
    length, width = pitch

    def full(values, scale):
        out = np.full((n,) + values.shape[1:], np.nan)
        out[index] = values * scale
        return out

    ball_x = full(df["ball_x"].to_numpy(float), length)
    ball_y = full(df["ball_y"].to_numpy(float), width)
    ball_x[gt == GT_D] = np.nan
    ball_y[gt == GT_D] = np.nan
    px = full(df[["{}_x".format(p) for p in players]].to_numpy(float), length)
    period_full = np.zeros(n, dtype=np.int8)
    period_full[index] = period
    match = MatchTracking(
        n_frames=n, fps=int(round(dataset.metadata.frame_rate)),
        pitch_length=length, pitch_width=width,
        team_of_track=np.array(teams, dtype=np.int8),
        players_x=px, players_y=full(df[["{}_y".format(p) for p in players]].to_numpy(float), width),
        players_speed=np.full_like(px, np.nan),
        ball_x=ball_x, ball_y=ball_y, ball_speed=np.full(n, np.nan), period=period_full)
    return match, gt


def is_dfl(game: str) -> bool:
    return game.startswith("J03")


def load_bench_game(game: str):
    """(MatchTracking, gt) de un partido del banco: "1".."3" Metrica, "J03..." DFL."""
    return load_dfl_bench(game) if is_dfl(game) else load_metrica_bench(game)


def load_dfl_bench(match_id: str, cache_dir: str | None = None):
    """(MatchTracking, gt_codes) de un partido DFL/IDSSE con la GT de su tracking, con cache .npz."""
    from hackbarca.io.dfl import DEFAULT_CACHE as DFL_CACHE
    from hackbarca.io.dfl import dfl_paths

    cache_dir = cache_dir or DFL_CACHE
    cache = os.path.join(cache_dir, "bench_{}.npz".format(match_id))
    if not os.path.exists(cache):
        from kloppy import sportec

        paths = dfl_paths(match_id, cache_dir)
        dataset = sportec.load_tracking(meta_data=paths["meta"], raw_data=paths["tracking"],
                                        coordinates="metrica", only_alive=False)
        match, gt = dfl_to_bench(dataset)
        np.savez_compressed(cache, gt=gt, **{k: v for k, v in vars(match).items()})
    return _load_cache(cache)
