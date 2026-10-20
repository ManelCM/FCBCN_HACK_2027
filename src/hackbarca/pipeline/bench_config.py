"""Configuracion fija del banco externo (ADR-0006). Cambiarla invalida las comparaciones previas."""

from hackbarca.io.degrade import DegradeParams

BENCH_GAMES = ("1", "2", "3")

# Calibrado (rejilla de 54 puntos) para que el baseline oficial en el banco degradado reproduzca
# los 8 numeros publicados en EVALUATION.md para el partido completo real.
BENCH_DEGRADE = DegradeParams(
    ball_drop_rate=0.20,
    ball_gap_mean=60.0,
    dead_ball_visible=0.4,
    occlusion_boost=4.0,
)
