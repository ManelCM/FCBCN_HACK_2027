# 0008 — Protocolo pre-registrado para aceptar un cambio de modelo

- **Estado**: Propuesta
- **Fecha**: 2026-10-21
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
No hay leaderboard y la GT completa es privada. Vamos a probar decenas de variantes (features,
modelo secuencial, HSMM, balón fusionado) sobre un banco de pocos partidos. Sin una regla fijada
de antemano acabaríamos eligiendo la variante con más suerte, es decir, sobreajustando al banco.

## Decisión
Un cambio sustituye al modelo vigente solo si cumple las tres condiciones:
1. Se evalúa **dejando un partido fuera** en el banco vigente (v2 con 3 partidos; v3 con 10 cuando
   exista), con los mismos pliegues, semillas y búsqueda de decodificación que el modelo vigente.
2. Mejora el **spellF1 agregado en ≥ 0,01**.
3. Mejora en **la mayoría de los partidos retenidos**: los 3 de 3 en v2; al menos 7 de 10 en v3.

Además:
- Los 5 min públicos **nunca deciden**: solo sirven para detectar que algo se ha roto. Si un
  cambio baja ahí más de 0,10, se investiga antes de aceptarlo.
- Cada evaluación se registra en `docs/experimentos.md`, también las que no pasan.
- Si cambia el banco (v2 → v3), se re-evalúan el modelo vigente y el candidato en el nuevo banco
  antes de comparar.

## Justificación
Con ~1.100 posesiones en v2, el ruido del spellF1 agregado ronda ±0,02. Exigir a la vez una mejora
agregada y una mejora consistente por partido reduce mucho los falsos positivos sin necesidad de
un test formal en cada iteración.

## Consecuencias
- Algunas mejoras reales pero pequeñas se rechazarán. Es preferible a quedarnos con ruido.
- El umbral se revisa cuando exista v3 (más partidos, menos ruido).
