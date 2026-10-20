"""Configuracion fija del banco externo (ADR-0006). Cambiarla invalida las comparaciones previas."""

from hackbarca.io.degrade import DegradeParams

BENCH_GAMES = ("1", "2", "3")

# v1 (E002-E004): rejilla contra los 8 numeros publicados del baseline. Huecos de balon cortos
# (mediana 50 frames), muy distintos del ATD real (mediana 268, atd_stats E005).
BENCH_DEGRADE_V1 = DegradeParams(
    ball_drop_rate=0.20,
    ball_gap_mean=60.0,
    dead_ball_visible=0.4,
    occlusion_boost=4.0,
)

# v2 (E006): huecos largos como el ATD. Objetivos: 8 numeros publicados + balon ausente 0,407 +
# huecos p50/p90 268/845 + 56 % de D dentro de los huecos largos (GT publica).
BENCH_DEGRADE_V2 = DegradeParams(
    ball_model="long",
    gap_median=268.0,
    dead_weight=100.0,
    ball_drop_rate=0.45,
    dead_ball_visible=0.55,
    occlusion_boost=1.0,
)

BENCH_DEGRADE = BENCH_DEGRADE_V2
