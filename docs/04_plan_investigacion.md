# Plan de investigación y mejora — La MasIA, fase 1 (desde el 23 oct hasta la congelación del 14 nov)

## 1. Contexto: por qué este plan
Manel pide parar, revisar todo como haría un investigador, apoyarse en lo que ya existe y pensar el problema como un entrenador:
- **balón**: dónde está, si se mueve o está quieto, hacia dónde avanza;
- **jugadores**: quién aprieta, quién achica, quién se desmarca, quién sprinta, qué hace el portero;
- **bloques**: cuánto se abren y se cierran, cómo de juntos están, hasta dónde suben o bajan las líneas;
- **posiciones y estructura**: forma con y sin balón, cambios de formación, quién va adelantado.

Además quiere usar el **comentario en inglés** del broadcast para confirmar situaciones y posesiones, y que las features se organicen **por familias**, no como v8/v9/v10.

## 2. Revisión crítica de lo hecho

### Qué ha mejorado (banco, dejando un partido fuera)
| paso | efecto | evidencia |
|---|---|---|
| Viterbi sobre probabilidades por frame | 0,107 → 0,555 | E007 |
| LightGBM frente a reglas (L1) | +0,15 a +0,19 | E008 |
| Más capacidad (127 hojas) | +0,023 | E010 |
| Toques con umbral relativo al partido | +0,054 | E014 |
| Apilado (capa 2 con contexto de 1-10 s) | +0,029 | E018 |
| Medir con la posesión «controlada» | el examen pasa de 0,542 a 0,6245; el modelo no cambia | E020-E021 |
| Entrenamiento determinista + artefacto final | `La MasIA.csv` reproducible bit a bit (sha `252716b0…`) | E022-E024 |
| Banco de 10 partidos (DFL), calibrado sin tocar el simulador | fiabilidad para elegir | E027 |

### Qué no funcionó (descartado, con número)
- Smart Tagging: +0,008.
- Toques con umbral fijo (v2): espejismo de +0,083, porque el balón real está filtrado.
- Empuje de bloque como nivel (v4): +0,009.
- Aleatorización del dominio.
- Reentrenar con la GT controlada: −0,002.
- Rellenar el balón con el broadcast: 7,5 % de los huecos.
- YOLO genérico: 24 % a < 2 m.
- Silbato por procesado de señal: sin señal.
- Humo en los 5 min públicos: 0,588 → 0,513, pero la banda de ruido es ±0,38, así que no dice nada.

### Pendiente de medir
- Las features nuevas: cambio de posesión, colocación y tácticas.
- La referencia del banco v3. Está corriendo; se cortó una vez por memoria y ya está resuelto con la caché.

### Dónde fallamos más y por qué (lectura de entrenador sobre E020 y E028)
1. **Robos en juego: el 79 % de las posesiones perdidas.**
   - Las intercepciones solo se encuentran el 53 % de las veces y los balones sueltos el 58 %. Tras un envío perdido acertamos el 58 %.
   - **Por qué:** el modelo mira sobre todo la cercanía y los toques, pero en el ATD los toques casi no se ven y el balón se pierde justo en el choque.
   - Lo que un entrenador ve en los 1-3 s siguientes a un robo:
     - el ganador se gira hacia la portería rival, abre los extremos, sube los laterales y lanza el primer pase o la conducción hacia delante;
     - el perdedor o contrapresiona (los 2-4 más cercanos esprintan al balón) o repliega todo el bloque.
   - Hoy eso solo entra de forma débil.
2. **Huecos largos sin balón: 38 % de los frames, acierto 0,78.**
   - **Por qué:** sin balón, el modelo confunde «sigue el juego» con «balón parado» el 22 % de las veces.
   - Un balón parado tiene una colocación típica (barrera, área llena, alguien en la banda o en el centro). El juego vivo tiene forma y velocidad de juego.
3. **Duelos, aéreos y despejes.**
   - Sin altura del balón.
   - En un despeje, la posesión sigue siendo de quien despeja hasta que el rival **controla**, pero el modelo cambia antes, por cercanía: en vuelo acierta el 0,40.
4. **Infrasegmentación** (738 posesiones predichas frente a 868 reales).
   - El coste de cambio del Viterbi es el mismo en todos los frames: no sabe dónde es probable que haya un cambio.

## 3. Lo que ya existe (no reinventamos la rueda)
| trabajo | qué aporta | cómo lo usamos |
|---|---|---|
| **PathCRF** (Kim et al., 2026, arXiv 2602.12080, código público) | Posesión **sin balón**, solo con trayectorias de jugadores en **los mismos 7 partidos DFL**: 92,4 % de acierto de equipo por frame. Atención por conjuntos (social intra-equipo + global) + CRF con transiciones **dependientes del contexto** + Viterbi. Lecciones: lo social es lo que más aporta, 5 fps basta, las transiciones dinámicas corrigen el número de eventos | Modelo profundo como 2.ª fuente de probabilidades (§4.3) y **costes de cambio dinámicos** (§4.2) |
| **Ball Radar** (Kim et al., KDD 2023, código público, entrenable con Metrica) y **TranSPORTmer** (2024) | Inferir **dónde está el balón** a partir de los jugadores | Tarea auxiliar del modelo profundo: predecir la posición del balón en los huecos |
| **Pressing Intensity** (Bekkers, 2025, `unravelsports`) | Presión como probabilidad de interceptar: tiempo que tarda cada defensor en llegar a cada atacante (velocidad, dirección, reacción 0,7 s) | Familia «jugadores»: quién aprieta y cuánto |
| **Contrapresión** (Bauer y Anzer, 2021, DFL) | Detectan la contrapresión con 134 features y XGBoost (AUC 0,87) | Firmas de la transición tras pérdida |
| **Compacidad por fase** (longitud, anchura, longitud/anchura, superficie, stretch index) | Con balón los equipos son **más anchos, profundos y extensos** (AMISCO, LJMU) | Familia «bloques» |
| **Formaciones** (Bialkowski 2016, Shaw y Glickman 2019, SoccerCPD 2022, Bauer et al. 2023) | Forma distinta con y sin balón; asignación de roles; detectar cambios de formación | Familia «estructura»: parecido a la forma con balón o sin balón del propio partido |
| **SoccerNet-Echoes** (2024) | Whisper sobre comentarios de partidos de tele. Lección: falla con los nombres propios, conviene darle una lista | Comentario en inglés con `initial_prompt` de plantillas (§4.4) |
| **SkillCorner open data** | Tracking de tele a 10 fps con poseedor por frame | Posible 4.ª fuente de banco, solo si sobra tiempo |

## 4. Propuesta

### 4.1 Features por familias (una sola estructura, sin v8/v9)
- Se sustituye `version` por **grupos con nombre** en `src/hackbarca/features/frame_features.py`: `FEATURE_GROUPS = {"base", "touch", "ball", "players", "blocks", "structure", "restart"}`.
- El artefacto guarda la lista de grupos en `l3_decode.json`.
- `version=3` sigue funcionando como alias de `{"base", "touch"}`, para reproducir el modelo vigente.
- Lo ya programado (v5, colocación y tácticas en `features/tactical.py`) se recoloca en estos grupos.
- Todo es colectivo y sin identidades, con pares `_h`/`_a`, y cada señal incluye su **cambio después−antes** en 1 s y 3 s, porque los robos se ven en el cambio, no en el nivel.

| familia | features (idea de entrenador → medida) |
|---|---|
| **balón** | Estado y velocidad. Quieto o no, y cuánto lleva así. **Vuelo largo**: rápido y lejos de todos, así que la posesión se mantiene. **Sentido de progresión** respecto a cada portería, antes y después. Distancia al más cercano de cada equipo. Toques relativos (ya existen) |
| **jugadores** | **Quién aprieta**: presión al balón tipo *pressing intensity* simplificada (tiempo de llegada al balón o al probable poseedor con velocidad y dirección) y nº de rivales a < 5 m. **Contrapresión**: sprints hacia el balón en los 3 s siguientes. **Desmarques**: ritmo al que los atacantes se separan de su marcador. Receptor probable (ya existe). **Portero**: el más retrasado, pero solo si está a < 25 m de su portería (corrige el desajuste de 0,2-0,4 visto en el ATD); su distancia a la línea de gol y si vuelve |
| **bloques** | Por equipo: **longitud, anchura, superficie** (envolvente convexa), **stretch index** y relación longitud/anchura. **Altura de la línea defensiva** (4 más retrasados sin el portero) y de la delantera. Distancia entre líneas. Centroide de cada equipo y **distancia entre los dos bloques**. **«Achicar»**: la longitud baja deprisa. **«Apretar»**: la línea defensiva y el bloque suben a la vez. Todo con su cambio antes/después |
| **estructura** | **Jugadores por detrás del balón** de cada equipo (el que defiende tiene más). **Quién va adelantado**: nº en campo rival y por delante del balón. **Forma con balón y sin balón del propio partido**: plantillas medias de cada equipo calculadas con los frames en que el modelo base está seguro (≥ 0,9), sin etiquetas, como el umbral de toques; la feature es a cuál se parece más la forma actual. **Cambio de formación**: distancia entre la forma actual y la de hace 3 s |
| **reanudación** | La colocación ya programada: jugadores quietos, área llena, banda, área pequeña, barrera, saque de centro |

**Comprobación de dominio** (`pipeline/feature_shift.py`, ya existe): cada feature nueva tiene que tener un desajuste real/banco < 0,25. Si no, se corrige o se quita.

### 4.2 Costes de cambio dinámicos (lección de PathCRF)
- Un **detector de cambio** (LightGBM) aprende la probabilidad de que haya un cambio H↔A o vivo↔muerto a ±0,5 s, con todas las familias.
- El Viterbi usa `coste_cambio(t) = base · exp(−α · logit(p_cambio(t)))`: es más barato cambiar donde las señales de robo se disparan.
- Ataca a la vez los robos no vistos y la infrasegmentación.
- Cambio pequeño en `src/hackbarca/decode/viterbi.py`: matriz de transición por frame, con la misma interfaz. α y la base se eligen dejando un partido fuera.

### 4.3 Modelo profundo inspirado en PathCRF / Ball Radar (2.ª fuente)
- `src/hackbarca/models/seq.py` (torch, grupo `vision` ya instalado, GPU para entrenar).
- **Entrada**: por frame (a 5 fps), el **conjunto** de jugadores visibles: posición, velocidad, equipo y la distancia al balón si se ve. Es atención sobre conjuntos, así que no necesita identidades estables, lo que encaja con los slots fragmentados del ATD.
- **Arquitectura**: atención intra-equipo y global → embedding por frame → BiGRU en ventanas de 10 s.
- **Salidas**: P(H/A/D) por frame y, como tarea auxiliar, **la posición del balón**, aprendida donde el banco limpio la tiene. Así aprende dónde está el balón dentro de los huecos.
- **Uso**: sus probabilidades y el balón inferido entran como features en la capa 2 del apilado, en ensamblado con LightGBM.
- **Reproducibilidad**:
  - semillas, `torch.use_deterministic_algorithms` y **predicción en CPU**;
  - pesos versionados (pocos MB);
  - la entrega no necesita GPU.
- Es el paso más caro (2-3 días), así que va detrás de 4.1-4.2.

### 4.4 Comentario en inglés (confirmar situaciones y posesiones)
- `src/hackbarca/audio/commentary.py`: **faster-whisper** en local, large-v3 int8 en la RTX 3060, unos 20-30 min para 1 h 50 de audio.
  - `initial_prompt` con las plantillas públicas de Barça y Athletic 2024-25, para que acierte los nombres.
  - Llevado al timeline táctico con el desfase por parte (E026).
- **Señales**:
  - **equipo mencionado**: nombre de jugador → equipo;
  - **eventos**: «foul», «free kick», «corner», «throw-in», «offside», «goal kick», «goal», «wins it back», «intercepted», «gives it away».

  Se sacan en ventanas de tiempo con un retraso de 1-3 s.
- **Privacidad (ADR-0004)**: la transcripción se queda en `data/derived/`. **El agente nunca la lee**; los scripts solo devuelven agregados.
- **Validación sin banco**, porque los partidos públicos no tienen audio:
  - acuerdo entre el equipo mencionado y el modelo donde el modelo está seguro;
  - acuerdo entre los eventos del comentario y los parones predichos.
  - Solo si el acuerdo es alto (≥ 85 %) se usa, y **solo** para corregir el equipo o el balón muerto en tramos largos donde el modelo duda. Nunca para colocar fronteras.
  - La regla y su único umbral se fijan con ese acuerdo, nunca con la GT.
- **Q6** (uso del audio del broadcast): hay que enviarla. Mientras no haya respuesta, va como opción desactivada en la entrega.

### 4.5 Cosas baratas al final
- Ensamblado de 3-5 semillas.
- Calibración isotónica de la capa final.
- Reelegir el Viterbi en el banco v3.

## 5. Protocolo de evaluación (sin cambios en el espíritu de ADR-0008)
- **Banco v3**: 10 partidos dejando uno fuera; Metrica con T = 1,5 s y DFL con T = 2 s; entrenamiento con la GT original.
- **Se acepta** si sube ≥ 0,01 y mejora en ≥ 7 de 10 partidos.
- Siempre se informa además de:
  - % de cambios H↔A con frontera a ≤ 2 s (hoy 49 %);
  - acierto tras un envío perdido (hoy 0,58);
  - acierto en huecos largos (hoy 0,78);
  - nº de posesiones frente a la GT.
- **Evaluaciones, de una en una** (~1 h cada una):
  - E-ref: referencia (corriendo);
  - E-fam: todas las familias;
  - E-ablación: quitar una familia cada vez (solo si E-fam mejora), para quedarnos con las que aportan y explicarlo en el documento técnico;
  - E-dyn: costes dinámicos;
  - E-deep: modelo profundo en el apilado;
  - E-ens: semillas.
- **Comprobaciones en el reto real** (solo agregados): desajuste de features, `make_submission` dos veces con el mismo sha, humo con la GT pública (solo para detectar roturas).

## 6. Calendario
| fechas | trabajo |
|---|---|
| 23-25 oct | Grupos de features (refactor + familias nuevas + portero corregido) → comprobación de dominio → E-fam → ablación |
| 26-28 oct | Detector de cambio + costes dinámicos (E-dyn) |
| 29 oct-2 nov | Modelo profundo (E-deep) |
| 3-6 nov | Comentario: transcripción local, acuerdo con el modelo, regla conservadora |
| 7-9 nov | Ensamblado, calibración, análisis de fallos final |
| 10-14 nov | Artefacto final con los 10 partidos, clon limpio con el mismo sha, documento técnico. **Congelación el 14** |
| 15-16 nov | Repo público (PR a `main` aprobada por un compañero), PDF y Form |

## 7. Ficheros principales
- `src/hackbarca/features/frame_features.py` (grupos) y `src/hackbarca/features/tactical.py` (jugadores, bloques, estructura y reanudación).
- `src/hackbarca/decode/viterbi.py` (coste por frame).
- `src/hackbarca/models/l3.py` (capa 2 con entradas extra).
- Nuevos: `src/hackbarca/models/change.py`, `src/hackbarca/models/seq.py`, `src/hackbarca/audio/commentary.py`.
- `src/hackbarca/pipeline/tune_l3.py` (`--groups`, `--dynamic-switch`, `--deep`) y `train_l3_final.py`.
- Se reutilizan la caché de features de `tune_l3`, `feature_shift.py`, `failure_analysis.py` y `scripts/run_aggregate.sh`.

## 8. Qué necesito de vosotros
- **Enviar el email** (Q2/Q4/Q5/Q6). Q6 decide si el comentario puede entrar en la entrega.
- **Invitar a los compañeros** en GitHub y que alguien apruebe la PR antes del 15 nov.
- Confirmar la inscripción de los 3 (10 nov) y subir el Form.

## 9. Verificación
- Tests sintéticos por familia, uno por cada idea de entrenador: robo simulado, córner, línea que sube, portero que vuelve, bloque que se cierra.
- Detector de cambio y Viterbi dinámico con una señal sintética.
- Modelo profundo: determinismo (dos entrenamientos con la misma salida en CPU).
- Comentario: con un audio sintético o público, nunca el del reto delante del agente.
- `uv run pytest`; regresión v2 = 0,6245 con `--groups base,touch`.
- Banco v3 con el protocolo; `make_submission` dos veces con el mismo sha; clon limpio el 14 nov.

## Fuentes
- PathCRF: https://arxiv.org/abs/2602.12080 · código https://github.com/hyunsungkim-ds/pathcrf
- Ball Radar: https://arxiv.org/abs/2306.08206 · https://github.com/pientist/ballradar · TranSPORTmer https://arxiv.org/abs/2410.17785
- Pressing Intensity: https://arxiv.org/abs/2501.04712 · https://unravelsports.readthedocs.io
- Contrapresión: https://link.springer.com/article/10.1007/s10618-021-00763-7
- Formaciones: https://arxiv.org/abs/2502.03342 · https://doi.org/10.3233/JSA-220620 · SoccerCPD https://paperswithcode.com/paper/soccercpd-formation-and-role-change-point
- Compacidad por fase: https://researchonline.ljmu.ac.uk/19091/1/2022guangzezhangmphil.pdf
- SoccerNet-Echoes: https://arxiv.org/abs/2405.07354
- SkillCorner open data: https://github.com/skillcorner/opendata
