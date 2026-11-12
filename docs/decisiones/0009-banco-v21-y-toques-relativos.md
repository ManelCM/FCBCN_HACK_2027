# 0009 — Banco v2.1 (balón filtrado como el ATD) y features de toque relativas al partido (v3)

- **Estado**: Propuesta
- **Fecha**: 2026-10-21
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
Las features de toque (v2, E012) dieron +0,083 en el banco v2. Pero con datos reales
(`atd_stats`, E013) apareció un desajuste de dominio: el ATD detecta 0,02-0,04 toques/s y el banco
~0,49. La causa: **la trayectoria del balón del ATD está filtrada**:

| | ATD real | banco v2 | Metrica limpio |
|---|---|---|---|
| residuo de posición frente a la media móvil de 9 frames | 0,04 m | 0,31 m | — |
| dv p50/p90/p99 (m/s) | 1,8 / 5,6 / 11,8 | 3,2 / 8,6 / 21 | 0,4 / 8,6 / 21 |
| velocidad p50/p90/p99 (m/s) | 5,1 / 14,3 / 24,5 | 5,6 / 15,1 / 24,1 | — |
| velocidad justo antes de perder el balón, p50 | 2,0 m/s | 5,7 m/s | — |

El banco v2 metía ruido blanco, justo lo contrario. Un umbral fijo de 9 m/s casi nunca salta en
el ATD, y en los 5 min públicos L3 v2 daba 10 posesiones de 25 s (GT: 17 de 14 s).

## Opciones consideradas
1. **Quedarse con v2 y el banco v2**: no representan el balón real. Descartada.
2. **Solo recalibrar el simulador**: depende de acertar el filtro exacto del tracker de Metrica.
3. **Solo features relativas**: robustas, pero el banco seguiría sin parecerse al reto.
4. **Ambas**: banco v2.1 con el balón filtrado + features v3 con el umbral relativo al partido.
   Elegida.

## Decisión
- **Banco v2.1** (`bench_config.BENCH_DEGRADE_V2_1`): v2 más suavizado gaussiano de la
  trayectoria por tramos (σ = 10 frames), un error lento correlado AR(1) (0,8 m, τ = 75 frames) y
  ruido blanco de 0,02 m. Elegido con una rejilla de 32 puntos solo sobre la dinámica del balón,
  con objetivos de `atd_stats`: dv, velocidad, residuo y tasa de toques. Resultado: dv
  2,2/5,0/8,6, 0,042 toques/s.
- **Features v3**: el umbral de toque de cada partido se fija para 0,4 toques/s de balón visible
  (los eventos con balón de Metrica son ~0,42/s), y el dv entra como rango percentil dentro del
  partido. Es una normalización sin etiquetas.
- v2 queda en cuarentena; v3 es la candidata (E014).

## Justificación
E014, banco v2.1, dejando un partido fuera: v1 0,465 · v2 0,519 · **v3 0,519**. Las features de
toque mejoran los 3 partidos también con el balón realista. v3 empata con v2 dentro del banco
pero no depende de la escala del dv, que es justo lo que falló.

## Consecuencias
- Todo número anterior a v2.1 (E006-E012) es optimista y no se compara con los nuevos.
- La segunda calibración candidata (σ = 6, 0,3 m, τ = 25) se usa como prueba de robustez
  (`tune_l3 --eval-shift`).
- Si se consiguen más agregados del ATD (p. ej. la autocorrelación de la velocidad), se revisa
  v2.1.
