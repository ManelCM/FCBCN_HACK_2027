# 0018 — Guardarraíl de forma en el reto: agregados de nuestra predicción como aprobado o suspenso

- **Estado**: Propuesta (aprobada por Manel el 25 oct; falta la revisión de las otras dos personas)
- **Fecha**: 2026-10-26 (**umbrales escritos antes de entrenar ningún modelo del banco v3.1**)
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
- E041-E043 (`docs/investigacion/04_analisis_fallos_v3.md` §5-6): en el reto el modelo de la entrega mete unas
  seis veces más D cortos que en el banco (0,68 por minuto frente a 0,09), el 90 % con el balón visible. Causa:
  el simulador del banco borra el balón lento del juego vivo.
- Inyectado en el banco, el defecto cuesta −0,088 de spellF1 (E040).
- **El banco no puede medir este fallo**: su degradación no lo genera. Por eso el protocolo de ADR-0008 por sí
  solo no puede aceptar ni rechazar el arreglo (M1, `docs/05_plan_mejora.md`).
- No hay GT del reto ni leaderboard. La ventana pública (5 min, 17 posesiones) tiene un ruido de ±0,38 y no
  puede elegir.

## Opciones consideradas
1. **Solo el banco** (ADR-0008). Es la opción más conservadora, pero no ve el defecto más caro.
2. **Elegir parámetros mirando los agregados del reto**, por ejemplo el umbral de una regla para minimizar los D
   cortos. Riesgo de ajustar a ciegas al partido de test: descartada.
3. **Guardarraíl de aprobado o suspenso, pre-registrado**: unos pocos agregados de forma de nuestra predicción
   en el reto, sin GT, con umbrales fijados antes de ver ningún resultado. Solo decide si un modelo ya elegido
   en el banco puede pasar a la entrega. Elegida.

## Decisión
Un modelo candidato a la entrega, elegido antes en el banco con el protocolo de ADR-0008, **solo sustituye al
vigente si `pipeline/real_vs_bench.py` sobre su `La MasIA.csv` cumple a la vez**:

| medida (partido completo, frames puntuados) | umbral | vigente E070 |
|---|---|---|
| tramos D < 5 s por minuto | **≤ 0,25** | 0,68 ✗ |
| tramos D por minuto | **0,8-1,3** | 1,78 ✗ |
| cambios H↔A directos por minuto | **≥ 1,7** | 1,40 ✗ |
| % de frames D | **0,38-0,50** | 0,448 ✓ |

- Los umbrales salen del banco: GT de v3 y nuestra predicción en v3 (D cortos 0,09/0,12; tramos D 1,00/1,06;
  cambios directos 2,03/1,77; % de D 0,40/0,43), con un margen amplio. **No se cambian después de ver un
  resultado.** Si un candidato no pasa, se analiza con `short_dead_report` y se arregla la causa, no el umbral.
- El guardarraíl **no ordena candidatos**: no se elige el que mejor lo cumple, sino el mejor del banco que lo
  cumple.
- Se ejecuta con `scripts/run_aggregate.sh` (ADR-0004: solo agregados).

## Justificación
- Los agregados son de **nuestra predicción**, no de la GT: no hay etiquetas del reto en juego.
- Es una comprobación de dominio como la del umbral de toques relativo al partido (ADR-0009) o `feature_shift`.
- Un aprobado o suspenso con cuatro umbrales fijados antes no tiene grados de libertad para sobreajustar.
- Las cuatro medidas atacan exactamente el defecto medido (E041) y su coste (E040).

## Consecuencias
- Para cambiar la entrega hacen falta las dos cosas: aprobar el protocolo del banco **y** el guardarraíl.
- Si ningún candidato lo pasa, se queda la entrega vigente.
- Se revisa si cambia la definición de la GT o si aparece otro defecto del reto. Una revisión de umbrales exige
  un ADR nuevo, justificado con el banco, nunca con el resultado de un candidato.
