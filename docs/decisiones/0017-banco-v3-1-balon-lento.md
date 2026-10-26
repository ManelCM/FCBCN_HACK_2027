# 0017 — Banco v3.1: balón del simulador recalibrado para que exista el balón lento en juego vivo

- **Estado**: Propuesta
- **Fecha**: 2026-10-26
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
- E041-E043 (`docs/investigacion/04_analisis_fallos_v3.md` §5-6): en el reto el modelo de la entrega mete unas
  seis veces más D cortos que en el banco, el 90 % con el balón visible. TreeSHAP los atribuye a `ball_speed` y
  `touch_rank`: el balón lento.
- Tramos de balón lento (< 1,25 m/s) ≥ 1 s por minuto de juego vivo:

  | fuente | tramos/min |
  |---|---|
  | banco limpio | 1,08 |
  | banco degradado v3 | **0,01** |
  | reto, vivo predicho | 0,55 |
  | ventana pública con GT | 1,86 |

  En el banco, un balón visible y lento solo existe en los parones, y el modelo aprende «lento = D».
- El defecto inyectado cuesta −0,088 (E040).

## Opciones consideradas
1. **Dejar v3** y parchear la salida (regla de D cortos): en el banco no ayuda (E040) y en el reto sería ajustar
   a ciegas.
2. **Recalibrar solo el balón del simulador** con objetivos medibles (E013 + balón lento) y volver a entrenar.
   Elegida.
3. **Quitar o normalizar `ball_speed`**: pierde una señal útil para los parones reales y no arregla el dominio.

## Decisión
- `pipeline/calibrate_ball.py` (E047, solo datos externos) busca en dos pasadas de 48 puntos los parámetros del
  balón que reproducen los objetivos del reto (`atd_stats --json`) y el balón lento del banco limpio.
  - **1.ª pasada** (solo suavizar el error correlado, `ball_corr_smooth` nuevo): el balón lento no vuelve
    (0,01/min). El ruido blanco final (`ball_noise` 0,02, ~0,04 m de salto por frame) corta cualquier racha.
  - **2.ª pasada** (con `ball_noise` en la rejilla): `ball_noise` 0,005, `ball_smooth_sigma` 6,
    `ball_corr_noise` 0,8, `ball_corr_tau` 75, `ball_corr_smooth` 10.
- **`BENCH_DEGRADE_V3_1`** en `pipeline/bench_config.py`; banco `v3_1` en `tune_l3`/`train_l3_final` (mismos 10
  partidos, misma GT controlada). El resto del simulador (huecos, jugadores) no cambia.

| medida | objetivo (reto / limpio) | v3 | **v3.1** |
|---|---|---|---|
| residuo frente a la media de 9 frames (m) | 0,044 | 0,108 | 0,040 |
| dv p50 / p90 / p99 (m/s) | 1,8 / 5,6 / 11,8 | 2,2 / 4,9 / 8,1 | 1,1 / 5,1 / 10,1 |
| velocidad p50 / p90 (m/s) | 5,1 / 14,3 | 5,0 / 12,4 | 4,6 / 13,1 |
| toques/s | 0,043 | 0,032 | 0,060 |
| velocidad antes de un hueco (m/s) | 2,0 | 0,8 | 0,7 |
| % de frames lentos en vivo | 8,8 % (limpio) | 3,2 % | 7,9 % |
| tramos lentos ≥ 1 s por minuto en vivo | 1,08 (limpio); reto ≥ 0,55 | 0,006 | **0,96** |
| pérdida de la rejilla | — | 160,2 | **20,5** |

## Justificación
- Los objetivos son agregados del ATD real (ADR-0004) o del tracking limpio externo. Nada sale de la GT del
  reto.
- La velocidad antes de un hueco no se arregla con el ruido: depende de dónde abre los huecos el simulador. Se
  deja anotada.

## Consecuencias
- **Las cifras de v3.1 no son comparables con las de v3**. Una referencia nueva de v3.1 sustituye a 0,7206.
- E048: el modelo E070 (entrenado en v3) evaluado en v3.1 tiene que mostrar el defecto. Si no lo muestra, la
  hipótesis está mal y se revisa antes de seguir.
- E049: reentrenar en v3.1. Se acepta con el protocolo (ADR-0008) frente a E048 y, para la entrega, también con
  el guardarraíl de ADR-0018.
- Tocar `degrade.py` cambia la clave de la caché de features: hay que recalcular la caché de los dos bancos.
