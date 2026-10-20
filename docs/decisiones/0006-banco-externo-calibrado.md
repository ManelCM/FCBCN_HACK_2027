# 0006 — Banco externo: Metrica sample-data degradado y calibrado con el baseline publicado

- **Estado**: Propuesta
- **Fecha**: 2026-10-20
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
No hay leaderboard y la GT del partido completo es **privada**: `run_evaluation.py --gt private`
solo funciona en la máquina de la organización. Solo podemos puntuar contra los 5 min públicos
(17 posesiones, que el propio repo oficial demuestra que invierten el orden de los modelos). Para
elegir modelos e hiperparámetros necesitamos un conjunto de validación propio (ADR-0005), y los
agentes no pueden tocar los datos del reto (ADR-0004).

Hallazgos que condicionan el diseño:
1. **DFL/IDSSE (figshare) bloquea la descarga automática**: responde `202` vacío o `403` a todo
   cliente que no sea un navegador. `hackbarca.io.dfl` queda listo pero hay que bajar los
   ficheros a mano a `external_data/dfl/<match_id>/{meta,event,tracking}.xml`.
2. **Metrica sample-data no trae los flags del tracking corregido** ("balón muerto", "equipo
   dueño") de los que sale la GT del reto: solo x,y. La GT hay que derivarla de sus eventos.
3. En el tracking corregido de Metrica, **que falte el balón es casi sinónimo de balón muerto**:
   falta en el 91-99 % de los frames D y solo en el 9-14 % de los vivos. Si el ATD automático
   se pareciera, "no hay balón" sería un detector de D casi gratis. Pero el baseline oficial
   tiene en el reto real un macroF1 que solo se reproduce si **el ATD sí ve el balón en buena
   parte de los parones** (ver calibración). Hay que confirmarlo con `atd_stats` (abajo).
4. El Smart Tagging solo trae 59 `SET PIECES` (5 faltas) para más de 100 parones reales:
   cubre como mucho la mitad de los D.

## Opciones consideradas
1. **Solo los 5 min públicos**: gratis, pero 17 posesiones y orden de modelos engañoso. Descartada.
2. **Banco limpio (Metrica sin degradar)**: el baseline da spellF1 0,10/0,36/0,41/0,30
   (raw/s25/s50/s125), muy lejos de lo publicado para el reto (0,049/0,200/0,243/0,224), y s25
   gana a s125, al revés que en el reto. No representa el problema real. Descartada.
3. **Banco degradado con parámetros a ojo**: no sabemos si se parece al ATD.
4. **Banco degradado calibrado contra los números publicados del baseline**: los 8 números de
   EVALUATION.md (spellF1 y macroF1 de 4 variantes) los calculó la organización sobre el partido
   real completo; son públicos y agregados. Elegida.

## Decisión
- Banco = Metrica sample-data games 1, 2 y 3 (~1.140 posesiones, 425 k frames evaluados).
- GT por frame derivada de eventos (`hackbarca.io.external.gt_from_events`): equipo del último
  evento de control (PASS, SET PIECE, SHOT, RECOVERY, CARRY, BALL LOST; **no** CHALLENGE, porque
  un duelo no es control); D desde que muere el balón (fin del evento previo, o la falta) hasta
  cada SET PIECE; X fuera de cada periodo.
- Degradación (`hackbarca.io.degrade`, semilla = índice del partido): borrado del balón en rachas,
  más probable cerca de jugadores; parte de los huecos de balón muerto pasan a visibles; tracks
  partidos en tracklets reasignados a 46 slots sin identidad; ruido gaussiano.
- Parámetros fijos en `hackbarca.pipeline.bench_config.BENCH_DEGRADE`, elegidos por rejilla
  (54 puntos) minimizando el error cuadrático contra los 8 números publicados + la tasa de balón
  ausente (41 %).
- Toda comparación de modelos se reporta como spellF1 **agregado, dejando un partido fuera**.

## Justificación
Baseline oficial en el banco calibrado frente al partido real (E001, E002):

| baseline | spellF1 banco | spellF1 real | macroF1 banco | macroF1 real |
|---|---|---|---|---|
| raw | 0,026 | 0,049 | 0,584 | 0,653 |
| s25 | 0,219 | 0,200 | 0,652 | 0,675 |
| s50 | 0,293 | 0,243 | 0,668 | 0,672 |
| s125 | 0,268 | 0,224 | 0,633 | 0,636 |

Reproduce el orden real (s50 > s125 > s25 > raw), que el banco limpio no reproducía, con el
macroF1 casi exacto. Balón ausente en frames evaluados: 36 % (real 41 %).

## Consecuencias
- El banco es algo más fácil que el reto en spellF1 (≈ +0,05): los números absolutos del banco
  **no** son predicciones del leaderboard; sirven para **ordenar** modelos.
- La GT por eventos no es la del tracking corregido: puede discrepar en fronteras finas (frames).
  Los ajustes de fronteras al frame hay que contrastarlos con los 5 min públicos solo como humo.
- Pendiente: (a) ejecutar `uv run python -m hackbarca.pipeline.atd_stats` sobre el ATD real
  (solo imprime agregados) y recalibrar si la distribución de huecos o de balón ausente
  vivo/muerto difiere; (b) refinar la rejilla (dos parámetros quedaron en el borde);
  (c) añadir DFL cuando se descargue a mano.
- Revisar esta decisión si `atd_stats` contradice el hallazgo 3.
