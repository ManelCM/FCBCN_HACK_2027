# 02 — Estrategia para ganar

_Borrador 2026-10-20 (actualizado con las bases oficiales). Lo revisamos los 3._

## Principio general
1. **Fase 1 = un número (spellF1), sin feedback.** Solo hay que quedar entre los **20 primeros**, pero sin leaderboard no sabremos dónde estamos hasta el final. Por eso apuntamos lo más alto posible con **validación propia fiable** (ADR-0005) y **nunca nos quedamos sin un fichero válido subido**.
2. **Fase 2 = reto abierto con jurado y mentoría.** La edición 2026 era así y valoraba la IA generativa y los *insights* tácticos accionables.
3. **Final = pitch.**

**La fase 1 no es trabajo tirado.** Un segmentador de posesión fiable sobre tracking automático es la base de cualquier análisis de la fase 2 (fases de juego, transiciones, presión tras pérdida). Hay que construirlo como una librería limpia y reutilizable.

**Lo que más pesa en spellF1** (ver investigación §1): no parpadear, fronteras precisas a < 0,5 s y `D` bien detectados.

## Fase 1 — plan (20 oct → 17 nov)

| Semana | Objetivo | Entregable |
|---|---|---|
| S1 (20-26 oct) | Inscripción de los 3 y descarga de datos · ADR-0004 aprobado · reproducir el baseline oficial (L0) · verificar los datasets externos · cargadores y visualizador | Primer fichero validado (L0/L1) **subido al Form** |
| S2 (27 oct-2 nov) | Simulador de degradación + evaluación con datos externos · detector vivo/muerto · toques · HMM (L2) | Board interno con spellF1 en externos |
| S3 (3-9 nov) | LightGBM + HSMM (L3) · ajuste fino de fronteras · análisis de errores por tipo | Candidato principal |
| S4 (10-17 nov) | Congelar el **15 nov** · reproducibilidad exacta desde un clone limpio · documento técnico · repo público | Envío final el 16 nov, no el 17 |

### Reparto inicial propuesto (rotable)
- **A — Datos e infraestructura**: cargadores (ATD, Smart Tagging, homografía), loaders de datos externos (DFL, Metrica sample), **simulador de degradación**, reproducibilidad (`uv`, comando único).
- **B — Reglas y evaluación**: harness de evaluación (scorer oficial sobre ventana pública + externos, bootstrap), detector **vivo/muerto**, detector de **toques**, L1/L2, análisis de errores.
- **C — Modelado**: features colectivas, LightGBM, **HMM/HSMM**, ajuste de fronteras, L3/L4.
- Todos: documento técnico (se escribe a medida que se avanza, no al final).

Interfaces que acordamos el día 1 (en un ADR):
- `load_match(...) -> FrameTable` (numpy/pandas, en metros, con NaN donde falta el dato) — común para el reto y los datasets externos.
- `predict(FrameTable) -> np.ndarray[str]` de longitud `n_frames`.
- `evaluate(pred, gt) -> dict` (envuelve `possession_metrics.evaluate`).

## Fase 2 — ideas para madurar (no decidido)
- **Asistente táctico en lenguaje natural** sobre la base de posesiones y fases (LLM + herramientas). **Ojo**: las reglas de LLM de la fase 1 (no enviar datos a LLMs en la nube) probablemente se mantengan, así que haría falta un LLM local o que el LLM solo vea métricas agregadas.
- **Presión tras pérdida ("5 segundos")**: tiempo hasta recuperar, jugadores implicados y zona. Es identidad del club.
- **Línea defensiva alta y trampa del fuera de juego**: muy característico del Barça reciente.
- **Valor de las decisiones** con *pitch control* / EPV.

## Riesgos
| Riesgo | Mitigación |
|---|---|
| **Violar la política de LLM** (un agente lee datos) → descalificación | ADR-0004, CLAUDE.md, hook que bloquee `data/` |
| Elegir mal el modelo por falta de feedback | ADR-0005 (validación externa) |
| Que el fichero no se pueda reproducir exactamente | Determinismo (semillas, versiones fijadas); en S4 probar desde un clone limpio en otra máquina |
| Quedarnos en 2 personas → fuera de fases 2-3 | Compromiso de los 3; un 4º de reserva |
| Error de formato o de alineación (off-by-one) | `validate_submission.py` sobre el fichero exacto antes de cada subida |
| Dejarlo para el último día | Fichero subido desde S1 y congelación el 15 nov |
