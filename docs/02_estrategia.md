# 02 — Estrategia para ganar

_Borrador 2026-10-20. Lo revisamos los 3 y lo convertimos en ADR cuando haya consenso._

## Principio general
Hay tres filtros distintos y cada uno premia una cosa diferente:
1. **Fase 1 = ranking automático.** Solo hay que estar en el **top 20**, no en el primer puesto. Premia la robustez y la rapidez: envío pronto, iteraciones medidas y nada de apuestas arriesgadas al final.
2. **Fase 2 = jurado técnico/táctico.** Premia una **pregunta táctica relevante para el Barça**, bien resuelta y explicada.
3. **Final = pitch.** Premia la historia, la demo en vivo y el impacto ("esto lo usaría el cuerpo técnico el lunes").

**Ventaja que hay que buscar:** que la fase 1 no sea trabajo tirado. La posesión por frame es la **base** de casi cualquier análisis táctico (fases de juego, transiciones, presión tras pérdida...). Si en la fase 1 construimos un pipeline limpio y reutilizable (carga → coordenadas de campo → posesión → segmentación en fases), en la fase 2 partimos con semanas de ventaja.

## Fase 1 — plan (20 oct → 17 nov)

| Semana | Objetivo |
|---|---|
| S1 (20-26 oct) | Datos descargados e inventariados; Q1-Q8 respondidas; carga y visualización funcionando; **primer envío L1** |
| S2 (27 oct-2 nov) | Validación local (etiquetas o etiquetado manual); L2 (eventos) y L3 (LightGBM + HMM) |
| S3 (3-9 nov) | Análisis de errores en transiciones; ensemble; **inscripción confirmada antes del 10 nov** |
| S4 (10-17 nov) | Congelar el mejor modelo; envío final sin prisas (no el último día) |

### Reparto inicial propuesto (rotable)
- **Persona A — Datos e infraestructura**: carga, homografía y coordenadas de campo, visualizador de frames/animación, formato de envío.
- **Persona B — Reglas y evaluación**: baseline L1/L2, evaluación local, análisis de errores y etiquetado manual.
- **Persona C — ML**: features, LightGBM + suavizado, y modelo secuencial si compensa.
Las tres líneas comparten **un solo módulo de evaluación** (`src/eval`), así que hay que acordar su interfaz el primer día.

## Fase 2 — ideas para empezar a madurar (no decidido)
La 1ª edición puso el foco en la **IA generativa** para convertir datos en *insights* tácticos. Ideas con gancho Barça:
- **Asistente táctico en lenguaje natural**: el analista pregunta "¿cuántas veces perdimos el balón en salida y en cuántos segundos recuperamos?" y un LLM con herramientas consulta una base de datos de posesiones, fases y eventos, y devuelve números + clips de vídeo + un diagrama. Demo potente para un pitch.
- **Presión tras pérdida ("regla de los 5 segundos")**: tiempo de recuperación, jugadores implicados y éxito por zona. Es identidad del club.
- **Línea defensiva alta y trampa del fuera de juego**: altura de la línea, sincronía y riesgo de espalda. Es muy característico del Barça reciente.
- **Valor de las decisiones** con *pitch control* / EPV: ¿había un pase mejor?
El criterio para elegir será: relevancia para el cuerpo técnico × viabilidad en 7 semanas × lo vistoso que sea en demo.

## Riesgos
| Riesgo | Mitigación |
|---|---|
| Quedarnos en 2 personas → descalificación de fases 2-3 | Compromiso explícito de los 3; contacto de un posible 4º miembro de reserva |
| Sobreajustar al leaderboard | Validación local por bloques y límite de envíos acordado |
| Desalineación entre agentes | CLAUDE.md, ADRs, LOG y PR obligatoria con revisión |
| Datos pesados (vídeo) | Fuera de git; carpeta compartida (Drive) para etiquetas manuales |
