import numpy as np

from hackbarca.io.load_match import AWAY, HOME, MatchTracking
from hackbarca.pipeline.kickoff_rule import apply_kickoff_dead, kickoff_frames


def _match(ball_x, ball_y):
    n = len(ball_x)
    px = np.zeros((n, 2))
    return MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0,
                         team_of_track=np.array([HOME, AWAY], dtype=np.int8), players_x=px, players_y=px.copy(),
                         players_speed=px.copy(), ball_x=ball_x, ball_y=ball_y, ball_speed=np.zeros(n))


def _event(a, b, typ="Kick Off (Start)", code="SET PIECES"):
    return {"code": code, "start_frame": a, "end_frame": b, "tags": {"Type": typ}}


def test_kickoff_from_tag_and_ball_movement():
    n = 1000
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    bx[600:], by[600:] = 52.5, 34.0              # balon quieto en el centro desde el frame 600...
    bx[640:] = 52.5 + np.arange(n - 640) * 0.5    # ...y sale en el 640 (pasa de 1 m en el 643)
    graded = [(100, 400), (500, 1000)]
    events = [_event(90, 215), _event(590, 715), _event(700, 800, typ="Throw-in")]
    kicks = kickoff_frames(events, _match(bx, by), graded)
    assert kicks == [90, 643]                     # 1.a parte: sin balon visible, inicio de la ventana
    codes = np.zeros(n, dtype=np.int8)
    out = apply_kickoff_dead(codes, kicks, graded)
    assert (out[500:643] == 2).all() and (out[643:] == 0).all()
    assert (out[100:400] == 0).all()              # la 1.a parte ya empieza en el saque


def test_no_tag_changes_nothing():
    n = 300
    m = _match(np.full(n, 50.0), np.full(n, 30.0))
    kicks = kickoff_frames([_event(10, 50, typ="Corner Kick")], m, [(0, 300)])
    assert kicks == [None]
    codes = np.ones(n, dtype=np.int8)
    assert np.array_equal(apply_kickoff_dead(codes, kicks, [(0, 300)]), codes)
