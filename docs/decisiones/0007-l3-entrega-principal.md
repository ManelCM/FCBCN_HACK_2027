# 0007 — L3 (LightGBM + Viterbi, sin Smart Tagging) como entrega principal; baseline de reserva

- **Estado**: Propuesta
- **Fecha**: 2026-10-20
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
No hay leaderboard y la GT completa es privada (ADR-0006). La única forma de elegir es el banco
externo calibrado v2, más el humo de los 5 min públicos para comprobar que el pipeline funciona.

## Opciones consideradas
1. **Baseline oficial s50**: banco v2 0,300. Seguro, pero es el suelo.
2. **L1** (reglas + Viterbi): banco v2 0,326 (+0,026, en el límite del ruido).
3. **L3 con Smart Tagging**: banco v2 0,486.
4. **L3 sin Smart Tagging**: banco v2 0,479.

## Decisión
- Entrega principal: **L3 sin Smart Tagging**.
  `uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name <equipo>`
- El modelo se **versiona** (`src/hackbarca/models/artifacts/l3_lgbm.txt`, 3,1 MB) junto con su
  decodificación (`l3_decode.json`). La entrega exige reproducir el fichero exacto, y no queremos
  depender de que LightGBM entrene bit a bit igual en otra máquina. Se regenera con
  `train_l3_final` (determinista: dos entrenamientos dan el mismo sha256 `fbce83ed…`).
- Reserva: el baseline s50 (`--model baseline`).

## Justificación
E008: L3 supera al baseline en +0,18 de spellF1 dejando un partido fuera en el banco v2, muy por
encima del ruido (~±0,02). El Smart Tagging aporta +0,008, dentro del ruido. En el banco las
etiquetas simuladas están perfectamente centradas en la reanudación, mientras que en el ATD solo
el 13 % de los huecos largos acaba en un SET PIECE (E006). Con la misma puntuación, la opción que
no depende de ese supuesto es más robusta.

## Consecuencias
- El banco es ~+0,05 más fácil que el reto en spellF1: esperamos bastante menos que 0,48 en el
  partido real, pero el orden frente al baseline debería mantenerse.
- L3 infra-segmenta (726 posesiones frente a 1.138) y sobreajusta (0,64 dentro de muestra frente
  a 0,48 fuera): las siguientes mejoras van por regularización, más partidos (DFL) y fronteras.
- Cambiar el modelo o la decodificación exige un experimento nuevo y regenerar el artefacto.
