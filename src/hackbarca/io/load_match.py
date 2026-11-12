"""Cargador comun de tracking: datos del reto (ATD) y datos externos.

Formato de salida (`MatchTracking`): posiciones en METROS, NaN donde falta el dato,
frames 0-indexados (video_frame = frame_id - 1, ver DATA.md / games.py del repo oficial).
Este modulo no abre ningun fichero bajo data/ por si mismo mas alla de lo que el
propio usuario le pida via `data_dir`/`MTAH_DATA_DIR` (ADR-0004: los agentes no leen
datos del reto; esto solo es la tuberia que ejecuta una persona o un test con datos
sinteticos).
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_OFFICIAL_REPO = os.path.join(_REPO_ROOT, "external", "more-than-a-hack-2027")
if _OFFICIAL_REPO not in sys.path:
    sys.path.insert(0, _OFFICIAL_REPO)

try:
    from games import GAME, GAMES, DEFAULT_SOURCE, require, source_frames  # noqa: E402
except ImportError as exc:  # pragma: no cover - solo si el submodulo no esta inicializado
    raise ImportError(
        "no se encuentra games.py del repo oficial en {}. "
        "Inicializa el submodulo: git submodule update --init".format(_OFFICIAL_REPO)
    ) from exc

HOME, AWAY = 0, 1

# Raw ATD: "<frame_id>:<player1_x,y,s>;<player2_x,y,s>;...:<ball_x,y,s>"
FIELD_SEP = ":"
GROUP_SEP = ";"
COORD_SEP = ","


@dataclass
class MatchTracking:
    """Tracking de un partido en METROS, indexado por video_frame (0-indexado).

    - `players_x/y/speed`: (n_frames, n_slots) float64, NaN donde no hay observacion.
    - `ball_x/y/speed`: (n_frames,) float64, NaN donde el balon no es visible.
    - `team_of_track`: (n_slots,) int8, HOME (0) o AWAY (1) segun el track slot.
    - `n_slots` no es nº de jugadores: son track slots (46 en el ATD tactico oficial),
      que pueden aparecer, desaparecer y cambiar de identidad (ver DATA.md).
    """

    n_frames: int
    fps: int
    pitch_length: float
    pitch_width: float
    team_of_track: np.ndarray
    players_x: np.ndarray
    players_y: np.ndarray
    players_speed: np.ndarray
    ball_x: np.ndarray
    ball_y: np.ndarray
    ball_speed: np.ndarray
    period: np.ndarray | None = None  # (n_frames,) 1/2, 0 sin dato; None si se desconoce

    @property
    def n_slots(self) -> int:
        return self.team_of_track.shape[0]


def team_of_each_track(meta_path: str) -> np.ndarray:
    """Array (n_slots,) HOME/AWAY por track slot, leido del XML de metadatos.

    El primer <Team> del fichero es el equipo local (convencion FIFA/EPTS, la que
    sigue kloppy en metadata.teams). Un slot es un tracklet, no una persona: no
    sigue a un jugador concreto durante todo el partido.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: se esperaban 2 equipos, encontrados {}".format(
            meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no hay entradas <Player>".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots],
                     dtype=np.int8)


def _parse_group(group: str) -> tuple[float, float, float]:
    """"x,y,s" normalizado (0-1) -> (x, y, s). "NaN,NaN,NaN" si falta la observacion."""
    first = group.index(COORD_SEP)
    second = group.index(COORD_SEP, first + 1)
    return (float(group[:first]), float(group[first + 1:second]), float(group[second + 1:]))


def parse_atd_raw(raw_path: str, n_frames: int, n_slots: int,
                   pitch_length: float, pitch_width: float):
    """Lee el fichero ATD crudo y devuelve arrays en METROS, indexados por video_frame.

    Devuelve (players_x, players_y, players_speed, ball_x, ball_y, ball_speed).
    El fichero cuenta frame_id desde 1; aqui se escribe en video_frame = frame_id - 1
    (ver DATA.md / starter/load_atd.py del repo oficial).
    """
    players_x = np.full((n_frames, n_slots), np.nan)
    players_y = np.full((n_frames, n_slots), np.nan)
    players_speed = np.full((n_frames, n_slots), np.nan)
    ball_x = np.full(n_frames, np.nan)
    ball_y = np.full(n_frames, np.nan)
    ball_speed = np.full(n_frames, np.nan)

    with open(raw_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            frame_part, players_part, ball_part = line.split(FIELD_SEP)
            video_frame = int(frame_part) - 1
            if not (0 <= video_frame < n_frames):
                continue

            for slot, group in enumerate(players_part.split(GROUP_SEP)):
                if slot >= n_slots:
                    break
                x, y, s = _parse_group(group)
                players_x[video_frame, slot] = x * pitch_length
                players_y[video_frame, slot] = y * pitch_width
                players_speed[video_frame, slot] = s

            bx, by, bs = _parse_group(ball_part)
            ball_x[video_frame] = bx * pitch_length
            ball_y[video_frame] = by * pitch_width
            ball_speed[video_frame] = bs

    return players_x, players_y, players_speed, ball_x, ball_y, ball_speed


def load_match(game: str = GAME, data_dir: str | None = None,
                source: str = DEFAULT_SOURCE) -> MatchTracking:
    """Carga el ATD de un partido como `MatchTracking`, en metros y 0-indexado.

    Usa `games.py` del repo oficial para resolver rutas (respeta MTAH_DATA_DIR).
    No hace falta kloppy: el parseo es directo sobre el fichero crudo, igual que
    `starter/baseline.py`, lo que permite probarlo con ficheros sinteticos del
    mismo formato sin depender de un entorno con kloppy instalado.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    n_frames = source_frames(source, game)
    pitch_length, pitch_width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]

    team_of_track = team_of_each_track(meta_path)
    n_slots = team_of_track.shape[0]

    (players_x, players_y, players_speed,
     ball_x, ball_y, ball_speed) = parse_atd_raw(
        raw_path, n_frames, n_slots, pitch_length, pitch_width)

    return MatchTracking(
        n_frames=n_frames,
        fps=fps,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        team_of_track=team_of_track,
        players_x=players_x,
        players_y=players_y,
        players_speed=players_speed,
        ball_x=ball_x,
        ball_y=ball_y,
        ball_speed=ball_speed,
    )
