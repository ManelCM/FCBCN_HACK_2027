# 0021 — Regla de D cortos (3 s) en la decodificación de la entrega

- **Estado**: Aceptada por Manel el 27 oct (falta la revisión de las otras dos personas)
- **Fecha**: 2026-10-27
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
- En el reto el modelo predice 0,53 D cortos (< 5 s entre dos tramos vivos) por minuto. La GT del banco tiene
  0,084 y nuestra predicción en el banco, 0,125 (E075c, E090b).
- El exceso cae sobre todo en **campo abierto**, donde no hay saques: 0,19/min en el reto frente a 0,004 en la
  GT del banco (E090b).
- Inyectado en el banco, ese exceso cuesta −0,056 de spellF1 al modelo E075 (E090).
- Una regla de decodificación (`failure_analysis.drop_short_dead`: un D de menos de k s entre dos tramos vivos
  toma la etiqueta del tramo anterior) no puede pasar ADR-0008 en el banco limpio, porque allí casi no hay
  D cortos que quitar. Es el mismo problema que justificó ADR-0018: el banco no ve el defecto del reto.

## Opciones consideradas
1. **No aplicar ninguna regla** y esperar a que L1 arregle el defecto desde el modelo. Riesgo: no llegar antes
   del 14 nov.
2. **Elegir k mirando el reto** (el que más baje los D cortos). Descartada: sería ajustar con el partido de
   test.
3. **k fijado solo con el banco y aceptado con un criterio doble**. Elegida, si el equipo lo aprueba:
   - **sin daño en el banco limpio**: no empeora el total, y ningún partido baja más de 0,005;
   - **mejora en el banco con el defecto del reto inyectado**.

   Después, el guardarraíl de ADR-0020.

## Decisión
- La entrega aplica la regla con **k = 3 s** (75 frames), la mayor que cumple el criterio sin daño:
  - banco limpio: +0,001, con 5 partidos mejor y 4 peor, y el peor partido a −0,004;
  - con 4 s, el peor partido baja −0,009.
- Con el defecto inyectado: 0,7038 → 0,7270 (+0,023).
- En el reto (E090):
  - D cortos 0,529 → 0,198/min;
  - tramos D 1,643 → 1,313/min;
  - cambios directos 1,586 → 1,775/min;
  - % de D 0,432 → 0,422;
  - mismo sha dos veces (`d976e575374b00d5`).

  Cumple ADR-0020 y 3 de los 4 objetivos de ADR-0018.
- Se configura en el artefacto (`l3_decode.json`, clave `short_dead_frames`), así que la entrega se reproduce
  con `make_submission` sin opciones.

## Justificación
- k no sale del reto, sino de un criterio fijado en el banco antes de mirar el resultado en el reto.
- La regla es de dominio: en el fútbol hay muy pocas interrupciones de menos de 3 s en mitad del campo con el
  mismo equipo antes y después. En la GT del banco hay 0,004 por minuto en campo abierto.
- Riesgo acotado: si un D corto del reto fuera real, la regla lo convierte en posesión del equipo anterior.
  Con k = 3 s solo afecta a tramos de menos de 3 s.

## Consecuencias
- Excepción explícita a ADR-0008 para reglas de decodificación que atacan un defecto que el banco no genera:
  hace falta **sin daño en el banco + mejora con el defecto inyectado + guardarraíl**.
- Cuando L1 arregle el defecto desde el modelo, se vuelve a medir si la regla sigue aportando.
