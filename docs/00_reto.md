# 00 — El reto: qué sabemos (bases oficiales)

_Última actualización: 2026-10-20. **Fuente de verdad: el repo oficial** https://github.com/metrica-sports/more-than-a-hack-2027 (commit `3de6b60`). Si este documento y el repo discrepan, manda el repo, y en concreto `evaluation/possession_metrics.py` para la métrica._

## Calendario

| Hito | Fecha |
|---|---|
| Datos publicados | 20 oct 2026 |
| Fin de inscripción | **10 nov 2026** |
| **Entrega fase 1** | **17 nov 2026** (Google Form; hace falta cuenta Google) |
| Resultados fase 1 | antes del 26 nov 2026 |
| Innovation Challenge (top 20) | 26 nov 2026 – 14 ene 2027 |
| Final (top 5, presencial) | 2 mar 2027, Talent Arena |

Equipos de **3 a 5 personas**. Con menos de 3 no se puede pasar a las fases 2 y 3.

## Fase 1: la tarea

Un partido: **FC Barcelona (H) vs Athletic Club (A)**, LaLiga 2024-25, jornada 2. Para **cada uno de los 199.319 frames** del vídeo táctico (25 fps) hay que dar una etiqueta:

| etiqueta | significado |
|---|---|
| `H` | posesión del local (Barça), balón vivo |
| `A` | posesión del visitante (Athletic), balón vivo |
| `D` | balón muerto |
| `X` | no contesto (o frame sin GT) |

Frames evaluados: 1ª parte **11.185–90.651** y 2ª parte **110.808–190.197** (158.857 frames, ~106 min). **Trampa:** la 2ª parte se evalúa desde 110.808 pero el saque no llega hasta **113.742**; esos 2.934 frames son `D`.

### Reglas de posesión (CHALLENGE.md)
- Hay **cambio de posesión** cuando un jugador rival obtiene **posesión controlada**. Con balón disputado o suelto, la posesión sigue siendo del último equipo que la tenía.
- **No es cambio**: un toque sin control, un desvío que no intercepta, un duelo aéreo ganado sin control posterior.
- Una posesión es una **racha máxima** de `H` o de `A`. `H H D D H H` son **dos** posesiones.
- **La GT es mecánica**: sale del *tracking corregido* de Metrica (`D` donde el tracking corregido marca balón muerto y, si no, el equipo que el tracking marca como poseedor). Sin suavizado ni retoques. La referencia real es la **GT pública de 5 min** (17 posesiones, 21 segmentos, duración media 14,2 s).

## Datos

| | **táctico** (referencia) | broadcast (opcional, **no sincronizado**) |
|---|---|---|
| vídeo | 199.319 frames, 25 fps, 1920×1080 | timeline propio, con cortes y repeticiones |
| tracking ATD | 199.319 registros | 164.488 |
| homografía | 100 % válida | 65,8 % válida |
| Smart Tagging | 638 instancias | — |
| balón visible | **56,9 %** de frames (59,3 % de los evaluados) | 50,5 % |

- **ATD** (formato EPTS de Metrica, XML + TXT, lo carga `kloppy==3.18.0`). Coordenadas normalizadas 0-1 (campo de 105,3 × 68,0 m). **46 *track slots*, sin identidad de jugador**: los slots aparecen, desaparecen y se intercambian; solo sabemos el equipo. Media de 20,5 objetos por frame. El dato que falta es `NaN`. `video_frame = frame_id - 1`.
- **El balón falta en el 41 % de los frames evaluados**, y de forma desproporcionada en los frames importantes (duelos, áreas, balones aéreos). **Es la dificultad central.**
- **Smart Tagging** (XML de SportsCode): 638 **fases** (no posesiones) con código, equipo y tipo. `video_frame = round(segundos * 25)`. Los *set pieces* (59) marcan reanudaciones, es decir, fin de balón muerto. Ojo: 36 tienen `Team = N/A`, **todos los saques de banda incluidos**. Es automático y tiene falsos positivos y negativos.
- **Homografía**: 3×3 por frame, de campo normalizado a píxeles. Sirve para proyectar el tracking sobre el vídeo.
- Mejorar o sustituir el tracking (detector propio, etc.) está **permitido**.

## Evaluación

- **Ranking por `spellF1`**: cada posesión (racha H/A) es un objeto. Se emparejan 1 a 1 con la GT por **IoU temporal** y se promedia la F1 en IoU 0,3/0,4/0,5/0,6/0,7. Rango [0, 1]; una respuesta constante saca 0.
  - 1 frame tarde: 1,0000 · 3 frames: 0,98 · 8 frames: 0,89 · **1 s tarde: 0,65**
  - Ruido: diferencias **< 0,04** no son significativas.
- **Desempate: `macroF1`** (F1 media por frame sobre H/A/D). Ruido < 0,01.
- **`X` en spellF1** = repetir la última etiqueta (*forward-fill* dentro de la parte). Abstenerse **dentro** de una posesión sale gratis; en una **transición** es caro.
- **Baseline a batir: spellF1 0,2433** (vecino más cercano + voto mayoritario de ±50 frames).
- **No hay leaderboard ni feedback.** Solo la GT de 5 min, que **no sirve para elegir entre versiones**: el propio repo demuestra que escoge mal la ventana del baseline.
- El scorer local es **exactamente** el oficial: `python evaluation/run_evaluation.py`.

## Entregables (un Google Form por equipo, editable hasta el deadline)
1. `<nombre_equipo>.csv`: `frame,label`, **199.319 filas**, pasado por `validate_submission.py`.
2. **Repositorio de código PÚBLICO** que **regenere exactamente el mismo fichero** (dependencias fijadas, comando único). Sin datos dentro.
3. **Documento técnico** `<nombre_equipo>.pdf` o `.md` (pocas páginas): enfoque y por qué, qué assets se usaron **y cuáles se descartaron**, reglas de posesión implementadas (balones sueltos, duelos, reanudaciones), limitaciones y cómo reproducirlo.

## Reglas críticas (RULES.md / FAQ / T&C)
- ✅ Cualquier método, lenguaje, modelo o dato externo, siempre que la organización pueda inspeccionarlo y el documento lo explique.
- ❌ **Etiquetar a mano el resultado.** Las etiquetas tienen que salir de un algoritmo. *Ambigüedad:* RULES.md permite "pequeñas cantidades de datos etiquetados a mano para **entrenar o validar**", pero el FAQ dice "Can I hand-label some frames? No". → **Preguntar en un issue antes de hacerlo.**
- ❌ Usar la GT privada.
- ❌ Entregar un fichero que el repo no pueda regenerar.
- ❌ Redistribuir los datos o publicar derivados que permitan reconstruirlos.
- ❌⚠️ **No subir ni enseñar a ningún LLM en la nube (ChatGPT, Claude, Gemini...) vídeos, datasets ni EXTRACTOS de ellos.** Nos afecta directamente porque trabajamos con Claude Code → ver **ADR-0004**.

## Preguntas abiertas (para la organización: issue en el repo oficial o email `hackathon.talentarena@mobileworldcapital.com`)
| # | Pregunta | Estado |
|---|---|---|
| Q1 | ¿Se permite etiquetar a mano unos minutos solo para **validar o entrenar** (RULES.md lo permite y el FAQ dice que no)? | Pendiente |
| Q2 | ¿Las métricas agregadas (spellF1 sobre la ventana pública, conteos de posesiones) cuentan como "extractos" que no se pueden enseñar a un LLM en la nube? | Pendiente |
| Q3 | ¿Podemos usar datos públicos de Metrica (sample-data) para entrenar? (RULES dice que sí a datos externos; confirmar) | Probablemente sí |
| Q4 | ¿Se puede versionar en el repo público un fichero **derivado** del vídeo (posiciones del balón detectadas por nuestro modelo) para que la entrega se reproduzca sin GPU? (ADR-0010) | Pendiente: **enviar** |
| Q5 | ¿Se pueden usar las etiquetas de la GT pública de 5 min para **entrenar** (no solo para evaluar)? Hoy no lo hacemos. | Pendiente |

**Borrador para enviar (issue en el repo oficial, en inglés):**

> Hi! Three quick questions to make sure we stay within the rules:
> 1. *Aggregates and LLMs.* We never share challenge data with cloud LLMs. Do aggregate statistics computed locally (e.g. the spellF1 of our own submission on the public window, or the percentage of frames with a visible ball) count as "extracts" under the T&C?
> 2. *Derived files in the public repo.* If we run a ball detector locally on the tactical video, may we commit the resulting per-frame ball positions to our public repository, so that the submission can be reproduced without a GPU? The detector code and the command that regenerates the file would be included.
> 3. *Public ground truth for training.* May the labels of the public 5-minute ground truth be used for training, or only for validation?
>
> Thanks!

## Fase 2 (top 20): retos abiertos sobre 10 partidos, con mentoría. Fase 3 (top 5): pitch ante jurado.
