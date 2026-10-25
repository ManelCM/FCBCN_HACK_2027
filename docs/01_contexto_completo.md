# Contexto completo del proyecto, desde cero

> Para quien llega sin saber nada: compañeros del equipo o un agente en una sesión nueva. Resume
> el reto, los datos, cómo validamos, cómo funciona el modelo, dónde fallamos y qué viene. Las
> cifras salen de `docs/experimentos.md` (E001-E037) y de los ADR; ante cualquier duda, mandan esos
> ficheros. Versión explicada y con gráficos: `docs/informe/La_MasIA_informe_estado_2026-10-24.pdf`.
> Actualizado: 2026-10-25 (análisis de fallos v3 y plan de mejora).

## 1. El reto

- **More Than A Hack 2027**, fase 1 (Metrica Sports). **Pasan 20 equipos.**
- Somos **«La MasIA»** (3 personas). **Entrega: 17 nov 2026; congelamos el 14.**
- Entrada: un partido, **Barça–Athletic**. Vídeo táctico a 25 fps: **199.319 frames**, de los que se
  puntúan 158.857.
- Salida: una etiqueta por frame.

| etiqueta | significado |
|---|---|
| **H** | posesión del Barça (local) |
| **A** | posesión del Athletic |
| **D** | balón muerto |
| **X** | sin respuesta; fuera de las partes no cuenta |

- Entregables: `La MasIA.csv`, `La MasIA.pdf` (documento técnico) y el repo **público** que regenera
  **el mismo CSV byte a byte** con un comando.

### La métrica: spellF1 (lee esto antes de proponer nada)

- **No puntúa frames, puntúa posesiones.** Una posesión es una racha máxima de H o de A; un D en medio
  la corta («H D H» son dos).
- Una posesión predicha acierta si **se solapa (IoU) con una real** por encima de un umbral. Se promedia
  en IoU 0,3-0,7.
- Consecuencias:
  - **no ver un robo cuesta 2-3 posesiones**;
  - **parpadear es letal**;
  - **llegar 1 s tarde a un cambio cuesta ~35 %**;
  - acertar frames no basta: el baseline oficial tiene macroF1 0,67 y spellF1 solo **0,243**.

### Qué cuenta como posesión (CHALLENGE.md)

- Cambia cuando el rival gana el balón **controlado**. Un toque, un desvío o un duelo sin control **no**
  cuentan.
- La GT sale del tracking corregido de forma mecánica. La referencia es la GT pública de 5 min:
  17 posesiones, 14,2 s de media, 17,6 % de menos de 3 s.

### Reglas que nos condicionan

1. **Los agentes de IA no ven los datos del reto** (ADR-0004).
   - El agente escribe el código, el código corre en local y solo vuelven agregados.
   - Desde el 22 oct, con autorización de Manel, el agente ejecuta los scripts de solo agregados a través
     de `scripts/run_aggregate.sh`, que oculta la salida si el script falla. **Pendiente de que el equipo
     lo apruebe en el ADR.**
2. **No hay leaderboard.** La GT completa es privada.
3. **Prohibido etiquetar a mano**, usar otras fuentes de posesión de este partido o subir los datos al
   repo.

## 2. Los datos

| fuente | qué es | estado | uso hoy |
|---|---|---|---|
| **ATD táctico** | posición de jugadores y balón por frame, sin identidades (46 slots) | ✅ | **base de todo** |
| **Vídeo táctico** | 1080p, homografía válida en todos los frames; audio mudo | ✅ | YOLO genérico **descartado** (24 % a < 2 m, E025); YOLO afinado con pseudoetiquetas encuentra el balón en el 55,9 % de los huecos, pero sobre todo en parones y pegado a jugadores: **rechazado** (E061-E069, ADR-0016) |
| **Broadcast** | tele con su ATD; no sincronizada; **audio con comentarista en inglés** | ✅ | como balón **no sirve** (7,5 % de los huecos, E023); alineado con un desfase por parte (E026); silbato sin señal (E029); comentario transcrito, solo como comprobación (E033) |
| **Smart Tagging** | eventos automáticos | ✅ | **descartado** (+0,008, ADR-0007) |
| **GT pública 5 min** | 17 posesiones | ✅ | solo prueba de humo y forma de la definición; **nunca elige modelos** |
| **Metrica (externo)** | 3 partidos con tracking y eventos | ✅ | banco (GT de eventos) |
| **DFL (externo)** | 7 partidos de la Bundesliga con flags de posesión en el tracking | ✅ desde el 23 oct | banco (GT del tracking, como la del reto) |

**Problemas del ATD:**
- **Falta el balón en el 40,7 %** de los frames puntuados, sobre todo en ~134 huecos de más de 4 s.
- La trayectoria del balón está filtrada: casi no se ven los toques.
- En el ATD real los equipos salen ~5 m más cortos que en el banco (estilo o jugadores no detectados).
  Por eso las medidas de forma van relativas al partido.

Rellenar el balón con otras fuentes no funciona: el balón falta en las mismas jugadas para todos. Los
huecos se atacan **sin balón**, con lo que hacen los 22 jugadores.

## 3. Cómo validamos sin leaderboard (ADR-0005, 0006, 0008, 0009, 0012, 0014)

- **Banco v3 = 3 Metrica + 7 DFL**, con el tracking limpio **estropeado para parecerse al ATD**: huecos
  largos de balón, balón suavizado, 46 slots sin identidad y ruido.
- **Calibración**: el baseline oficial reproduce en el banco el orden y la escala de los 4 números
  publicados (DFL, sin tocar nada: 0,025 / 0,203 / 0,265 / 0,236 frente a 0,049 / 0,200 / 0,243 / 0,224),
  y además coinciden los agregados del ATD real.
- **Posesión controlada**: un robo solo cuenta si el nuevo equipo conserva el balón ≥ T s (tras un parón
  siempre). Metrica usa T = 1,5 s y DFL T = 2 s. T se elige por la forma de la distribución (17,5 % de
  posesiones de menos de 3 s, como la GT pública), nunca por la nota. Se entrena con la GT original y se
  evalúa con la controlada.
- **Protocolo**:
  - siempre dejando un partido fuera;
  - un cambio entra si sube **≥ 0,01** y mejora en **≥ 7 de 10** partidos;
  - antes, comprobación de dominio de las features nuevas con `feature_shift`.
- **Límites**:
  - el banco es ~0,02-0,04 más fácil que el reto: sirve para ordenar modelos, no para predecir la nota;
  - la GT del banco aproxima la del reto, pero no es idéntica.

## 4. El modelo

1. **Features por frame**, todo colectivo y sin identidades.
   - Versión 3 (modelo de la entrega actual): unas 65 features de cercanía, toques relativos, receptor,
     velocidad colectiva, forma y contexto de los huecos.
   - **Familias** (ADR-0015, modelo de la entrega): balón, jugadores, bloques, estructura y reanudación (§6).
2. **Capa 1: LightGBM** (127 hojas, 400 rondas) → P(H), P(A), P(D). Entrena con equipos intercambiados y
   la predicción se simetriza.
3. **Capa 2 (apilado, ADR-0011)**: LightGBM que ve las probabilidades de la capa 1 en ventanas de 1-10 s.
4. **Viterbi** por parte con coste por cambio (1 / 20 / 10 / 10). Sin retraso.
5. **Formato**: X fuera de las partes, validador oficial, sha256.
   - El modelo va versionado en `src/hackbarca/models/artifacts/` y el entrenamiento es determinista bit a
     bit (E024).
   - `make_submission` admite también, si el artefacto lo pide, el modelo profundo (predicción en numpy) y
     el Viterbi dinámico. Ninguno de los dos se usa: E036 y E037.

**Entrega actual** (E070): `La MasIA.csv`, sha256 `9602959c4ec05bd6`, 336 posesiones, modelo de las cinco
familias (regenerado dos veces con el mismo sha). Ventana pública: spellF1 0,651 y macroF1 0,803.

## 5. Resultados

| banco | modelo | spellF1 | nota |
|---|---|---|---|
| v2 (3 Metrica, GT controlada) | apilado v3 | 0,6245 | entregado en el CSV actual |
| **v3 (10 partidos)** | apilado v3 (referencia) | **0,6470** | E030 |
| v3 | + modelo profundo promediado | 0,6570 | E034 (pasaba sobre la referencia) |
| **v3** | **apilado + familias** | **0,7185** | **E035/E070: modelo de la entrega** |
| v3 | ablaciones por familias | 0,7065-0,7192 | E038-E067: se quedan las cinco |
| v3 | familias + relleno de balón con YOLO (simulación fiel) | 0,7262 | E069: no pasa el protocolo (E064 daba 0,7484 con una simulación optimista) |
| v3 | familias + modelo profundo | 0,7172 / 0,7134 | E037: rechazado |
| v3 | familias + coste de cambio dinámico | igual (nunca se elige) | E036: rechazado |

**Lo que no funcionó** (con motivo en `experimentos.md`):
- Smart Tagging;
- toques con umbral fijo (espejismo por el balón filtrado);
- empuje de bloque como nivel;
- aleatorización del dominio;
- entrenar con la GT controlada;
- broadcast como balón;
- YOLO genérico;
- silbato;
- comentario dentro del modelo;
- coste de cambio dinámico;
- modelo profundo sobre las familias;
- relleno de balón con YOLO afinado (en el reto convertía parones en juego vivo).

**Estimación orientativa en el reto**: banco − ~0,04 → **~0,62-0,68** con el candidato, frente al 0,243 del
baseline. Muy incierta (±0,05 o más).

## 6. Dónde fallábamos y qué hemos hecho (E020, E028, E035)

1. **Robos en juego que no vemos: 79 % de las posesiones perdidas.** Solo la mitad de los cambios H↔A
   tenían una frontera a ≤ 2 s, y tras un pase perdido acertábamos el 58 %. En cambio, los cambios de juego
   y los balones largos que alguien controla se aciertan al 93-95 %. **Por qué**: el modelo miraba cercanía
   y toques, y en el ATD el balón falta justo en el choque.
2. **Huecos largos sin balón** (38 % de los frames): acierto 0,78 frente a 0,92 con balón; el juego vivo
   salía D el 22 %.
3. **Duelos, aéreos y despejes** (sin altura del balón).
4. **Infrasegmentación**: 2.298 posesiones predichas frente a 2.885 en el banco v3.

**Respuesta: features como las mira un entrenador** (ADR-0015, `features/tactical.py`). Cada señal lleva su
cambio entre el segundo anterior y el posterior:
- **balón**: sentido de progresión, vuelo largo, quieto;
- **jugadores**: quién aprieta, sprints, desmarques, portero;
- **bloques**: longitud, anchura, superficie, stretch, altura de líneas, distancia entre líneas, adelanto
  entre bloques, relativos al partido;
- **estructura**: jugadores por detrás del balón y en campo rival, forma propia abierta o cerrada, cambio de
  forma;
- **reanudación**: colocación de balón parado.

Resultado: **0,7185 (+0,072, 9/10)** y 2.669 posesiones. Ablación (25 oct): todas las familias aportan
algo salvo bloques y estructura por separado, que son redundantes entre sí (sin las dos, −0,012). **Cautelas**:
- las features del portero, importantes y con un desajuste real/banco de 0,22;
- posibles artefactos del simulador en «balón» (explica la caída de J03WR9, E060);
- humo en el reto hecho: 0,651 en la ventana pública (banda de ruido grande).

### Análisis de fallos del modelo de la entrega (25 oct, E039-E046, `docs/investigacion/04_analisis_fallos_v3.md`)
- **Banco v3: 0,7206** con la decodificación de la entrega.
- El margen está en **ver los robos cortos** (cambios H↔A no vistos +0,146, la mayoría con el balón visible) y en
  **no inventar cambios** (+0,090). El Viterbi está agotado (+0,001) y las fronteras de lo que ya vemos, casi
  (+0,015).
- **J03WR9 falla en las fronteras** (F1@0,7 0,50), no en los huecos.
- **En el reto, un defecto que el banco no ve**: unas seis veces más D cortos (0,68 por minuto frente a 0,09), el
  90 % con el balón visible, que parten posesiones y sustituyen robos.
  - **Causa**: el simulador del banco borra el balón lento del juego vivo (error correlado sin suavizar, unos
    4 m/s aparentes), y el modelo aprendió que balón lento es parón (TreeSHAP: `ball_speed`, `touch_rank`).
  - **Coste**: inyectado en el banco a la tasa del reto, **−0,088** (0,721 → 0,633). Es el fallo más caro.
- **Puertas**:
  - audio con PANNs: sin señal;
  - YOLO como evidencia de equipo: fiable (92 %), pero con poco recorrido hoy;
  - compacidad: ya explotada; tras un robo el nivel engaña y manda el cambio.

## 7. Qué viene (**`docs/05_plan_mejora.md`**, que sustituye a la tabla de `docs/03_plan_ganar.md` cuando se apruebe)

1. **M1**: banco v3.1 con el balón lento realista + guardarraíl de forma en el reto (26-28 oct).
2. **M2**: señal de control del balón para los robos cortos (features de control; si no basta, poseedor
   individual tipo PathCRF).
3. **M3**: más partidos de entrenamiento (SkillCorner y PFF), con curva de aprendizaje antes de invertir.
4. **M4**: semillas. M5/M6 (Ball Radar, YOLO como evidencia) si queda tiempo.
5. Reproducibilidad cruzada en otra máquina (sha `9602959c4ec05bd6`): persona C, urgente.
6. Congelar el 14 nov: clon limpio, documento técnico, repo público, formulario.

## 8. Dónde está cada cosa

- **Informe de estado explicado (PDF)**: `docs/informe/`. La presentación de `docs/presentacion/` es del
  22 oct y ya no está al día.
- **Plan de investigación y estado del arte**: `docs/04_plan_investigacion.md` y, ampliado el 25 oct,
  `docs/investigacion/05_estado_del_arte_v2.md`.
- **Análisis de fallos del modelo actual**: `docs/investigacion/04_analisis_fallos_v3.md`. **Plan de mejora**:
  `docs/05_plan_mejora.md`.
- **Prioridades y reparto**: `docs/03_plan_ganar.md`.
- **Decisiones**: `docs/decisiones/` (ADR 0001-0015, todas en *Propuesta* a falta de revisión).
- **Experimentos**: `docs/experimentos.md`. **Diario**: `docs/LOG.md`. **Datos**: `docs/datos.md`.
- **Código** (`src/hackbarca/`):
  - `io/`: carga, banco Metrica y DFL, degradación, broadcast;
  - `features/`: `frame_features.py` y `tactical.py`;
  - `models/`: l3, change, seq;
  - `decode/viterbi.py`;
  - `audio/`: silbato y comentario;
  - `vision/`;
  - `pipeline/`: tune_l3, tune_seq, train_l3_final, make_submission, failure_analysis, feature_shift,
    atd_stats.
- **Comandos clave**:
  - Evaluar en el banco v3: `uv run python -m hackbarca.pipeline.tune_l3 --bench v3 --no-tags --features touch+ball+players+blocks+structure+restart --stack --controlled 1.5 --controlled-dfl 2 --train-gt raw --jobs 2`
  - Desajuste real/banco de las features (datos reales, solo agregados): `scripts/run_aggregate.sh shift uv run python -m hackbarca.pipeline.feature_shift --only-new`
  - Análisis de fallos (banco v2): `uv run python -m hackbarca.pipeline.failure_analysis --controlled 1.5`
  - Entrega (datos reales): `scripts/run_aggregate.sh sub uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"`
