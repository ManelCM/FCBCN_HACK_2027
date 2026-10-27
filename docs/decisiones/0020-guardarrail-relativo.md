# 0020 — Guardarraíl relativo: un candidato entra si no empeora ninguna medida de forma del reto

- **Estado**: Aceptada por Manel el 27 oct (falta la revisión de las otras dos personas)
- **Fecha**: 2026-10-27
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐
- **Modifica**: ADR-0018. Sus umbrales absolutos dejan de ser requisito para cambiar la entrega y pasan a ser el
  objetivo de la línea L1 (`docs/06_plan_parones.md`).

## Contexto
- ADR-0018 exige, para cambiar la entrega, cuatro umbrales absolutos de forma en el reto. El más duro es
  tener ≤ 0,25 D cortos por minuto, cuando la entrega vigente (E070) tiene 0,68.
- Esos umbrales mezclan dos cosas:
  1. **no hacer daño**: no enviar un modelo que en el reto deforme más las posesiones que el vigente;
  2. **arreglar el defecto** de los D cortos.
- Hoy ningún candidato arregla el defecto, porque es un problema propio (L1). El guardarraíl absoluto bloquea
  por eso cualquier mejora, aunque mejore el banco y además acerque la forma del reto a la buena en las cuatro
  medidas.
- **Caso que lo motiva (se dice abiertamente): E075c**, el artefacto con el conductor por vídeo.
  - En el banco: 0,7692, y 0,7605 en la prueba de estrés (E075s), mejor en 10 de 10 en los dos casos.
  - En el reto, frente a E070:
    - D cortos 0,680 → 0,529/min;
    - tramos D 1,775 → 1,643/min;
    - cambios H↔A directos 1,397 → 1,586/min;
    - % de D 0,448 → 0,432.
  - Mejora en todas, pero no cruza los umbrales absolutos.
- ADR-0018 pedía no revisar sus umbrales a la vista del resultado de un candidato. Esta decisión **no relaja
  los umbrales para que pase un candidato concreto**: cambia qué pregunta responde el guardarraíl, y la nueva
  regla se aplica igual a todos los candidatos a partir de hoy.

## Opciones consideradas
1. **Mantener ADR-0018**: la entrega se queda con E070 hasta que L1 baje los D cortos de 0,25/min. Riesgo: si
   L1 no llega antes del 14 nov, perdemos todas las mejoras del banco (unos +0,04, estimado).
2. **Bajar los umbrales absolutos** (p. ej., D cortos ≤ 0,55). Descartada: sería elegir el número mirando al
   candidato.
3. **Guardarraíl relativo a la entrega vigente**: el candidato no puede empeorar ninguna de las cuatro medidas.
   Elegida.

## Decisión
Un candidato elegido en el banco con el protocolo de ADR-0008 sustituye a la entrega vigente si, en
`pipeline/real_vs_bench.py` sobre su CSV, cumple **a la vez**:

| medida (partido completo, frames puntuados) | regla frente a la entrega vigente |
|---|---|
| tramos D < 5 s por minuto | **≤ que la vigente** |
| tramos D por minuto | **distancia al intervalo 0,8-1,3 ≤ que la de la vigente** |
| cambios H↔A directos por minuto | **≥ que la vigente, o ≥ 1,7** |
| % de frames D | **0,38-0,50** (sin cambios) |

- Además: dos ejecuciones de `make_submission` con el mismo sha.
- Los umbrales de ADR-0018 siguen escritos como **objetivo** y se informan siempre junto a la regla relativa.
- El guardarraíl sigue sin ordenar candidatos: no se elige el que mejor lo cumple, sino el mejor del banco que
  lo cumple.
- Se ejecuta con `scripts/run_aggregate.sh` (ADR-0004, solo agregados). La ventana pública no interviene
  (ADR-0019).

## Justificación
- Mantiene la protección importante: en el reto, una entrega nueva nunca deforma más las posesiones que la
  vigente.
- E040 mide el coste del defecto en el banco: inyectar D cortos cuesta spellF1. Si un candidato tiene menos
  D cortos, no tiene más cambios inventados y mejora el banco, no hay motivo medido para esperar que en el reto
  sea peor.
- Sigue sin grados de libertad: la regla compara con la entrega vigente, no con un número elegido.

## Consecuencias
- E075 (`external_data/artefactos/e075/`, sha `9cb9f1d33262a71a`) cumple la regla y pasa a ser la entrega.
- Cada nueva entrega se convierte en la referencia: la forma del reto solo puede ir a mejor (efecto trinquete).
- L1 sigue siendo prioritaria: los D cortos (0,53/min) siguen lejos del objetivo de 0,25.
- Si las otras dos personas no lo aprueban en la revisión, la entrega vuelve a E070 (artefacto en
  `external_data/artefactos/e070/`).
