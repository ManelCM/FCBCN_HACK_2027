# 0014 — Banco v3: 3 partidos de Metrica + 7 de DFL, con la GT controlada en cada fuente

- **Estado**: Propuesta
- **Fecha**: 2026-10-23
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
Con 3 partidos (banco v2), el ruido del spellF1 agregado ronda ±0,02 y el protocolo (ADR-0008) exigía mejorar
en 3 de 3. Mejoras reales pero pequeñas no pasaban. Además, la GT de Metrica sale de eventos, no de tracking
corregido como la del reto. DFL/Sportec publica 7 partidos con tracking corregido que trae, por frame,
`BallStatus` (vivo/muerto) y `BallPossession` (equipo): la misma naturaleza que la GT del reto.

## Opciones consideradas
1. **Seguir con 3 partidos**: ruido alto y una sola fuente.
2. **Añadir DFL** (7 partidos, CC-BY 4.0), con la misma degradación y la misma métrica. Elegida.
3. **Añadir también SkillCorner open data** (tele, 10 fps): otra definición de posesión y otra frecuencia.
   Queda como opcional.

## Decisión
- **Banco v3 = Metrica 1-3 + DFL J03WMX, J03WN1, J03WPY, J03WOH, J03WQQ, J03WOY, J03WR9**
  (`bench_config.BENCH_GAMES_V3`, `tune_l3 --bench v3`).
- **GT de DFL**: D si el tracking marca el balón muerto; si no, el equipo de `BallPossession`
  (`external.gt_from_tracking`, `dfl_to_bench`). El balón se borra en los frames muertos para imitar el
  tracking limpio de Metrica, donde falta en el 91-99 % de los D. Así el mismo simulador se aplica igual.
- **Regla controlada por fuente** (ADR-0012): T = 1,5 s en Metrica y **T = 2 s en DFL**. El flag de DFL
  también cambia con cada toque (43,7 % de posesiones de menos de 3 s). Con T = 2 s queda en el 17,5 %,
  como la GT pública (17,6 %). T se elige por la forma de la distribución, nunca por la nota.
- **Entrenamiento con la GT original** (`--train-gt raw`) y **evaluación con la controlada**, como el
  modelo vigente (E021).
- **Protocolo**: mejora ≥ 0,01 agregada **y** en ≥ 7 de 10 partidos.
- **Descarga de DFL**: figshare bloquea las descargas automáticas. Se abren las URLs en el navegador y
  después se ejecuta `uv run python -m hackbarca.io.dfl --from-downloads <carpeta>`.

## Justificación
- El simulador calibrado en Metrica, aplicado a DFL **sin tocar ningún parámetro**, da 43,7 % de balón
  ausente (reto 41 %). El baseline sale 0,025 / 0,203 / 0,265 / 0,236, frente a 0,049 / 0,200 / 0,243 / 0,224
  en el reto: mismo orden y escala (E027).
- La referencia (features v3, apilado) en v3 da **0,6470** (E030). Entrenar con 9 partidos en vez de 2 ya
  sube los de Metrica ~0,04.

## Consecuencias
- Los números de v2 y v3 no se comparan. Toda evaluación nueva va en v3.
- Una evaluación tarda 1-2 h con `--jobs 2`. Hay caché de features en `external_data/features/` y conviene
  no lanzar más de una a la vez (memoria).
- El artefacto final se entrena con los 10 partidos (`train_l3_final --bench v3`).
- Se revisa si se añade SkillCorner o si la organización aclara la definición de posesión.
