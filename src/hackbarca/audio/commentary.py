"""Comentario en ingles del broadcast -> senales de equipo y de situacion (plan §4.4, E033).

Transcribe en local con faster-whisper (GPU) el audio del broadcast, lo lleva al timeline tactico
con el desfase por parte (E026) y saca dos senales:
- equipo mencionado: un apellido de la plantilla publica 2024-25 -> H o A;
- situaciones: foul, free kick, corner, throw-in, offside, goal kick, goal, penalty...
El comentarista va 1-3 s tarde y habla tambien de repeticiones: estas senales solo sirven para
confirmar o corregir tramos largos, nunca para colocar fronteras.

ADR-0004: la transcripcion es un derivado de los datos del reto. Se guarda en data/derived/ (que el
agente no lee) y el script solo imprime agregados (`commentary_report`).

    scripts/run_aggregate.sh comentario uv run --group audio --group vision python -m hackbarca.audio.commentary
"""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np

# Plantillas publicas 2024-25 (apellidos o nombres con que los nombra un comentarista).
HOME_NAMES = ("stegen", "pena", "szczesny", "kounde", "cubarsi", "inigo", "martinez", "balde", "araujo",
              "christensen", "garcia", "fort", "casado", "pedri", "frenkie", "jong", "gavi", "olmo",
              "fermin", "bernal", "lewandowski", "lewy", "raphinha", "lamine", "yamal", "ferran", "torres",
              "ansu", "fati", "gerard", "dani")
AWAY_NAMES = ("agirrezabala", "simon", "marcos", "vivian", "paredes", "yeray", "yuri", "berchiche", "lekue",
              "gorosabel", "boiro", "prados", "galarreta", "jauregizar", "vesga", "sancet", "berenguer",
              "williams", "nico", "inaki", "guruzeta", "djalo", "herrera", "serrano", "padilla", "unai")
HOME_TEAM = ("barcelona", "barca", "blaugrana")
AWAY_TEAM = ("athletic", "bilbao")
EVENTS = {
    "foul": ("foul", "fouled"), "free_kick": ("free kick", "free-kick", "freekick"),
    "corner": ("corner",), "throw_in": ("throw", "throw-in"), "offside": ("offside",),
    "goal_kick": ("goal kick", "goal-kick"), "goal": ("goal",), "penalty": ("penalty",),
    "card": ("yellow", "booked", "booking", "red card"),
    "turnover": ("wins it back", "intercept", "interception", "gives it away", "loses it", "tackle"),
}
PROMPT = ("FC Barcelona against Athletic Club. Lamine Yamal, Raphinha, Lewandowski, Pedri, Casado, Kounde, "
          "Cubarsi, Inigo Martinez, Balde, ter Stegen, Olmo, Fermin, Ferran Torres. Nico Williams, Inaki "
          "Williams, Sancet, Berenguer, Guruzeta, Vivian, Yeray, Yuri Berchiche, De Marcos, Agirrezabala, "
          "Prados, Ruiz de Galarreta, Jauregizar, Vesga.")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z\- ]", " ", text)


def word_signals(words):
    """[(tiempo_s, texto)] -> (menciones [(t, +1 local / -1 visitante)], eventos {tipo: [t]}).

    Los eventos de dos palabras se buscan en la ventana de la palabra actual y la siguiente."""
    mentions, events = [], {k: [] for k in EVENTS}
    toks = [(t, _norm(w).strip()) for t, w in words]
    for i, (t, w) in enumerate(toks):
        if not w:
            continue
        if w in HOME_NAMES:
            mentions.append((t, 1))
        elif w in AWAY_NAMES:
            mentions.append((t, -1))
        pair = w + " " + (toks[i + 1][1] if i + 1 < len(toks) else "")
        for kind, keys in EVENTS.items():
            if any(pair.startswith(k) for k in keys):
                events[kind].append(t)
    return mentions, events


def team_mentions(words):
    """Nombres de equipo (no de jugador): senal mas debil, se cuenta aparte."""
    return [(t, 1 if _norm(w).strip() in HOME_TEAM else -1) for t, w in words
            if _norm(w).strip() in HOME_TEAM + AWAY_TEAM]


def transcribe(path: str, model_size: str = "large-v3"):
    """[(tiempo_s, palabra)] del audio, en local con la GPU (DLLs de CUDA de torch)."""
    import torch
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
    from faster_whisper import WhisperModel

    from hackbarca.audio.whistle import read_audio
    model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    segments, _ = model.transcribe(read_audio(path), language="en", initial_prompt=PROMPT,
                                   word_timestamps=True, vad_filter=True, beam_size=5)
    return [(float(w.start), w.word) for seg in segments for w in (seg.words or [])]


def to_frames(times_s, offsets, fps: int = 25):
    """Segundos del broadcast -> frames tacticos (solo dentro de una parte)."""
    out = []
    for t in times_s:
        b = int(round(t * fps))
        for lo, hi, offset in offsets:
            if lo <= b < hi:
                out.append(b + offset)
                break
        else:
            out.append(-1)
    return np.array(out, dtype=np.int64)


def commentary_report(mention_frames, mention_team, event_frames: dict, pred: np.ndarray) -> dict:
    """Agregados: cuanto coinciden las menciones con el equipo predicho en los 2 s previos y los
    eventos de parada con un D predicho a +-5 s, frente al azar. Nada por frame."""
    rep = {"menciones_jugador": int(len(mention_frames)),
           "menciones_local": int(np.sum(mention_team == 1)),
           "menciones_visitante": int(np.sum(mention_team == -1))}
    agree = []
    for f, team in zip(mention_frames, mention_team):
        if f < 50:
            continue
        window = pred[f - 50:f]
        alive = window[window < 2]
        if len(alive) < 25:
            continue
        home = np.mean(alive == 0)
        agree.append((home > 0.5) == (team == 1))
    rep["acuerdo_equipo_mencionado"] = float(np.mean(agree)) if agree else float("nan")
    rep["n_menciones_comparables"] = len(agree)
    alive_pred = pred[pred < 2]
    rep["azar_cuota_local"] = float(np.mean(alive_pred == 0)) if len(alive_pred) else float("nan")
    rng = np.random.default_rng(0)
    graded = np.flatnonzero(pred < 3)
    base = np.mean([np.any(pred[max(0, f - 125):f + 125] == 2) for f in rng.choice(graded, 500)])
    rep["azar_D_a_5s"] = float(base)
    for kind, frames in event_frames.items():
        frames = [f for f in frames if f >= 0]
        if frames:
            hit = np.mean([np.any(pred[max(0, f - 125):f + 125] == 2) for f in frames])
            rep["evento_%s_n" % kind] = len(frames)
            rep["evento_%s_con_D_a_5s" % kind] = float(hit)
    return rep


def main(argv=None) -> int:
    import argparse

    import pandas as pd

    from hackbarca.audio.whistle import half_offsets
    from hackbarca.pipeline.atd_stats import GRADED

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args(argv)

    data_dir = os.environ.get("MTAH_DATA_DIR", "data")
    cache = os.path.join(data_dir, "derived", "commentary_words.npz")
    if not os.path.exists(cache):
        words = transcribe(os.path.join(data_dir, "broadcast", "BAR-ATH_broadcast.mp4"), args.model)
        np.savez_compressed(cache, t=np.array([t for t, _ in words]), w=np.array([w for _, w in words]))
    blob = np.load(cache)
    words = list(zip(blob["t"].tolist(), blob["w"].tolist()))
    offsets = half_offsets(np.load(os.path.join(data_dir, "derived", "broadcast_alignment.npz"))[
        "tactical_to_broadcast"], GRADED)
    mentions, events = word_signals(words)
    m_frames = to_frames([t for t, _ in mentions], offsets)
    m_team = np.array([s for _, s in mentions])
    keep = m_frames >= 0
    e_frames = {k: to_frames(v, offsets).tolist() for k, v in events.items()}
    np.savez_compressed(os.path.join(data_dir, "derived", "commentary_signals.npz"),
                        mention_frame=m_frames[keep], mention_team=m_team[keep],
                        **{"event_" + k: np.array([f for f in v if f >= 0], dtype=np.int64)
                           for k, v in e_frames.items()})
    labels = pd.read_csv(os.path.join("submissions", "La MasIA.csv"))["label"].to_numpy()
    pred = np.select([labels == "H", labels == "A", labels == "D"], [0, 1, 2], 3)
    print("palabras transcritas", len(words), "| menciones de equipo (nombre del club)", len(team_mentions(words)))
    for key, value in commentary_report(m_frames[keep], m_team[keep], e_frames, pred).items():
        print("{:<34} {}".format(key, round(value, 3) if isinstance(value, float) else value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
