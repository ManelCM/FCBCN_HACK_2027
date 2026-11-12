"""Configuracion fija del banco externo (ADR-0006). Cambiarla invalida las comparaciones previas."""

from dataclasses import replace

from hackbarca.io.degrade import DegradeParams

BENCH_GAMES = ("1", "2", "3")
# v3 (banco de 10): Metrica + DFL/IDSSE, cuya GT sale de los flags de su tracking (gt_from_tracking)
DFL_GAMES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
BENCH_GAMES_V3 = BENCH_GAMES + DFL_GAMES

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

# v2.1 (E013): como v2 pero con la trayectoria del balon filtrada como en el ATD real (residuo
# 0,04 m, patadas aplanadas: dv p50/p90/p99 1,8/5,6/11,8 y 0,04 toques/s, atd_stats). Rejilla de
# 32 puntos solo sobre la dinamica del balon.
BENCH_DEGRADE_V2_1 = replace(BENCH_DEGRADE_V2, ball_noise=0.02, ball_smooth_sigma=10.0,
                             ball_corr_noise=0.8, ball_corr_tau=75.0)

BENCH_DEGRADE = BENCH_DEGRADE_V2_1

# v3.1 (E047, ADR-0017): el ruido del balon de v2.1 borraba el balon lento del juego vivo (tramos
# lentos >= 1 s por minuto: limpio 1,08, v2.1 0,01, reto >= 0,55; E043) y el modelo aprendia "lento =
# muerto" (D cortos x6 en el reto, E041-E042). Rejilla de 48 puntos (calibrate_ball): ruido blanco
# casi nulo y error correlado suavizado; residuo 0,040, lento en vivo 7,9 % y 0,96 tramos/min.
BENCH_DEGRADE_V3_1 = replace(BENCH_DEGRADE_V2_1, ball_noise=0.005, ball_smooth_sigma=6.0,
                             ball_corr_noise=0.8, ball_corr_tau=75.0, ball_corr_smooth=10.0)
