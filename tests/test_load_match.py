"""Tests de load_match() con datos SINTETICOS (ADR-0004: nunca datos reales del reto)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "external", "more-than-a-hack-2027"))

from hackbarca.io.load_match import (  # noqa: E402
    load_match, parse_atd_raw, team_of_each_track,
)

META_XML = """<?xml version="1.0"?>
<main>
  <match>
    <Team id="home-team">
      <Name>Equipo Local</Name>
    </Team>
    <Team id="away-team">
      <Name>Equipo Visitante</Name>
    </Team>
    <Player id="p1" teamId="home-team"><ShirtNumber>1</ShirtNumber></Player>
    <Player id="p2" teamId="home-team"><ShirtNumber>2</ShirtNumber></Player>
    <Player id="p3" teamId="away-team"><ShirtNumber>3</ShirtNumber></Player>
    <Player id="p4" teamId="away-team"><ShirtNumber>4</ShirtNumber></Player>
  </match>
</main>
"""


def _raw_line(frame_id, groups, ball):
    return "{}:{}:{}\n".format(frame_id, ";".join(groups), ball)


@pytest.fixture
def synthetic_meta(tmp_path):
    path = tmp_path / "synthetic_atd.xml"
    path.write_text(META_XML)
    return str(path)


@pytest.fixture
def synthetic_raw(tmp_path):
    path = tmp_path / "synthetic_atd.txt"
    lines = [
        _raw_line(1, ["0.1,0.2,1.0", "0.3,0.4,2.0", "0.5,0.6,3.0", "0.7,0.8,4.0"],
                   "0.5,0.5,0.0"),
        _raw_line(2, ["0.11,0.21,1.1", "NaN,NaN,NaN", "0.51,0.61,3.1", "0.71,0.81,4.1"],
                   "NaN,NaN,NaN"),
        _raw_line(3, ["0.12,0.22,1.2", "0.32,0.42,2.2", "0.52,0.62,3.2", "0.72,0.82,4.2"],
                   "0.52,0.52,0.1"),
    ]
    path.write_text("".join(lines))
    return str(path)


def test_team_of_each_track(synthetic_meta):
    teams = team_of_each_track(synthetic_meta)
    np.testing.assert_array_equal(teams, [0, 0, 1, 1])


def test_team_of_each_track_rejects_bad_team_count(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<main><match><Team id='x'/><Player teamId='x'/></match></main>")
    with pytest.raises(ValueError):
        team_of_each_track(str(bad))


def test_parse_atd_raw_converts_to_metres_and_zero_indexes(synthetic_raw):
    pitch_length, pitch_width = 100.0, 50.0
    px, py, ps, bx, by, bs = parse_atd_raw(
        synthetic_raw, n_frames=3, n_slots=4,
        pitch_length=pitch_length, pitch_width=pitch_width)

    # frame_id=1 -> video_frame=0
    assert px[0, 0] == pytest.approx(0.1 * pitch_length)
    assert py[0, 0] == pytest.approx(0.2 * pitch_width)
    assert ps[0, 0] == pytest.approx(1.0)
    assert bx[0] == pytest.approx(0.5 * pitch_length)

    # frame_id=2 -> video_frame=1: balon y slot 1 ausentes -> NaN
    assert np.isnan(bx[1])
    assert np.isnan(py[1, 1])
    assert px[1, 0] == pytest.approx(0.11 * pitch_length)

    # frame_id=3 -> video_frame=2
    assert px[2, 3] == pytest.approx(0.72 * pitch_length)


def test_load_match_end_to_end(tmp_path, synthetic_meta, synthetic_raw, monkeypatch):
    import games

    data_dir = tmp_path / "data"
    folder = data_dir / "tactical"
    folder.mkdir(parents=True)
    (folder / "SYN-GAME_tactical_atd.xml").write_text(META_XML)
    (folder / "SYN-GAME_tactical_atd.txt").write_text(open(synthetic_raw).read())

    monkeypatch.setitem(games.GAMES, "SYN-GAME", {
        "n_video_frames": 3, "fps": 25, "pitch": (100.0, 50.0),
        "offsets": {1: 0}, "halves": {1: (1, 3)},
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 0, "second": 0, "frame": 0},
    })
    monkeypatch.setitem(games.SOURCES, "tactical", {
        **games.SOURCES["tactical"], "stem": "SYN-GAME_tactical", "n_video_frames": 3,
    })

    match = load_match(game="SYN-GAME", data_dir=str(data_dir))

    assert match.n_frames == 3
    assert match.n_slots == 4
    np.testing.assert_array_equal(match.team_of_track, [0, 0, 1, 1])
    assert match.players_x.shape == (3, 4)
    assert match.ball_x.shape == (3,)
    assert np.isnan(match.ball_x[1])
    assert match.ball_x[0] == pytest.approx(0.5 * 100.0)


def test_load_match_missing_data_raises_readable_error(tmp_path, monkeypatch):
    import games

    monkeypatch.setitem(games.GAMES, "SYN-EMPTY", {
        "n_video_frames": 3, "fps": 25, "pitch": (100.0, 50.0),
        "offsets": {1: 0}, "halves": {1: (1, 3)},
        "kickoff": {"minute": 0, "second": 0, "frame": 0},
        "kickoff_2nd": {"minute": 0, "second": 0, "frame": 0},
    })
    monkeypatch.setitem(games.SOURCES, "tactical", {
        **games.SOURCES["tactical"], "stem": "SYN-EMPTY_tactical", "n_video_frames": 3,
    })

    with pytest.raises(IOError):
        load_match(game="SYN-EMPTY", data_dir=str(tmp_path / "no-such-dir"))
