# Estado del arte ampliado: qué usa la gente y qué nos puede servir (25 oct)

> Búsqueda hecha por el agente (Manel) el 25 oct para el plan de mejora (`docs/05_plan_mejora.md`).
> Complementa `01_posesion_estado_del_arte.md` y la tabla del §3 de `docs/04_plan_investigacion.md`.
> Cada técnica lleva: qué hace, qué resultado publica, si hay código, qué datos pide, **a qué fallo
> nuestro ataca** (según `04_analisis_fallos_v3.md`) y un coste orientativo. Las cifras publicadas son de
> sus autores y en sus datos: **no son comparables con nuestro spellF1** salvo que se diga.

## Resumen en un minuto
1. **No hay nadie resolviendo exactamente nuestro problema** (posesión de equipo + balón muerto con
   tracking degradado y sin balón el 41 % del tiempo). Lo más cercano es **PathCRF** (KAIST, feb 2026), que
   usa *los mismos 7 partidos DFL* que nuestro banco y solo jugadores: 92,4 % de acierto de equipo por frame
   (solo juego vivo, tracking limpio, un partido de test).
2. Las tres ideas que más se repiten y que **no tenemos**:
   - **transiciones enmascaradas y dinámicas en el CRF/Viterbi** (PathCRF: lo que más sube la precisión);
   - **el poseedor individual como variable intermedia** (Ball Radar, PathCRF, «ball possessor recognition»):
     primero quién (jugador) y luego el equipo. Nosotros vamos directos a H/A/D;
   - **pérdidas y cabezas de frontera** contra la sobresegmentación (MS-TCN, ASRF: hasta +16 % de F1
     segmental en vídeo).
3. **Más datos abiertos** para entrenar: SkillCorner (10 partidos con fases de juego) y PFF FC Mundial 2022
   (64 partidos de tracking de tele + eventos, gratis con formulario). Nuestro E010 decía que el modelo
   sigue limitado por datos.
4. **Audio**: lo que se usa hoy son **redes preentrenadas en AudioSet** (PANNs/YAMNet, que traen las clases
   *Whistle*, *Cheering* y *Crowd*) y Whisper. Nuestro E029 probó silbato solo con procesado de señal.
5. **Vídeo**: los detectores de balón fallan justo donde nos falla el ATD (balón quieto, pegado a un
   jugador, botas como falsos positivos), lo mismo que vimos en E068. Los modelos temporales (TrackNet)
   ayudan con el balón rápido, no con el parado.

## 1. Posesión y eventos desde el tracking

| técnica | qué hace | resultado publicado | código | ataca | coste |
|---|---|---|---|---|---|
| **PathCRF** (Kim et al., 2026, arXiv 2602.12080) | Grafo de 22 jugadores + 4 nodos «fuera». En cada paso hay una arista activa: un lazo = un jugador controla; emisor→receptor = balón en vuelo. CRF con **transiciones dinámicas** (salen de los embeddings) y **máscara de transiciones imposibles** (−10⁴); Viterbi. Codificador social PPE-FPE (de Ball Radar) + temporal PE-SAB (de TranSPORTmer). 5 fps, ventanas de 10 s | En el partido DFL de test: acierto de arista 69,6 %, F1 de eventos 75,7 %, **equipo por frame 92,4 %**, % de posesión 62,6 frente a 62,1. El CRF estático enmascarado sobrepredice cambios (2.135 frente a 1.831 reales); el dinámico se queda en 1.961. Lo que más aporta: la máscara y el codificador social; el módulo temporal importa poco | [github.com/hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf) | robos no vistos, cambios sobrantes, «quién» | 3-5 días: adaptar a 46 slots sin identidad y a D, entrenar en el banco degradado |
| **Ball Radar** (Kim et al., KDD 2023, arXiv 2306.08206) | Set Transformer + Bi-LSTM jerárquico: primero **probabilidad de que cada jugador posea**, luego la trayectoria del balón. Pérdida de realismo (el balón solo gira cerca de un jugador) y posproceso por reglas | Metrica + GPS de la K League. Sin nada de balón: error 3,7 m y **equipo poseedor 85 %**. Con el 10 % del balón real: equipo 91,5 %; con el 20 %: error 2,1 m y **equipo 95 %** | [github.com/pientist/ballradar](https://github.com/pientist/ballradar) (entrenable en Metrica) | huecos largos (41 % de los frames) | 2-3 días |
| **TranSPORTmer** (Capellera et al., ACCV 2024, arXiv 2410.17785) | Transformer único para predecir e imputar jugadores, inferir el balón y **clasificar el estado del balón** (pase, posesión, sin control, fuera) con un agente CLS | Supera a los modelos específicos en inferencia e imputación del balón (cifras en el artículo) | No encontrado | huecos, D frente a vivo | Alto (reimplementar) |
| **Multi-Modal Soccer Scene Analysis** (arXiv 2512.19528) | Trayectorias + tipo de jugador + **recortes de imagen de cada jugador** (cámara táctica); preentrenamiento con máscara (*CropDrop*). Infiere balón, estado y poseedor | «Mejoras sustanciales» en las tres tareas sobre datos propios | No | poseedor en huecos | Muy alto: pide vídeo táctico etiquetado que no tenemos |
| **Vidal-Codina et al.** (2022, arXiv 2202.00804) | Reglas (árbol de decisión) sobre el poseedor por frame: pases, tiros, recepciones, intercepciones, balón parado | > 90 % de detección en la mayoría de eventos y proveedores (con balón) | No | firmas de reanudación y de pérdida | Bajo: ideas para features |
| **Fases de juego con GNN** (Renner et al., KIT/DFB, arXiv 2610.11571, oct 2026) | GNN (Delaunay + SEAConv) + LSTM; 7 fases por segundo | macroF1 0,80; **las transiciones son lo peor** (contraataque 0,54, transición defensiva 0,66): cortas y sensibles a 1 s de error | [gitlab.kit.edu](https://gitlab.kit.edu/vincent.renner/match_phase_detection) | confirma que la transición es lo difícil | — |

**Lección para nosotros**: los dos trabajos más cercanos (PathCRF y Ball Radar) **modelan al jugador
poseedor**, no al equipo. El equipo sale al agregar. Además, la **restricción de transiciones** (no se puede
pasar de «jugador A controla» a «jugador C del otro equipo controla» sin un vuelo o un duelo) es lo que más
reduce los cambios falsos. Nuestro Viterbi es de 3 estados con coste fijo: no tiene esa estructura.

## 2. Segmentación temporal (lo que mide spellF1)

| técnica | idea | evidencia | ataca |
|---|---|---|---|
| **MS-TCN / MS-TCN++** (Farha y Gall, CVPR 2019) | Etapas de convoluciones dilatadas que refinan la salida anterior + **pérdida de suavizado** (MSE truncado sobre log-probabilidades) | Reduce la sobresegmentación más que una KL | parpadeo, posesiones sobrantes |
| **ASRF** (Ishikawa et al., WACV 2021) | Dos ramas: clase por frame + **regresión de fronteras**. Las fronteras predichas refinan las clases | Hasta +16,1 % de F1 segmental y +13,7 % de edit en 50Salads, GTEA y Breakfast | fronteras desplazadas, robos tardíos |
| **ASFormer, DiffAct, LTContext** (2021-2023) | Transformers y difusión para segmentar acciones en vídeo | Estado del arte en F1 segmental | sobresegmentación |
| **HSMM / semi-CRF** | Duración explícita por estado (las posesiones < 1 s son raras) | Estándar en segmentación; sin trabajo específico de posesión | parpadeo, posesiones cortas falsas |

**Lección**: nuestras probabilidades por frame son buenas (macroF1 0,90 en el banco), pero el argmax sin
Viterbi da spellF1 0,36: **todo el valor está en cómo se convierte en segmentos**. Una cabeza de frontera
entrenada (ASRF) o un decodificador con duración aprendida atacan justo eso.

## 3. Inferir el balón desde los jugadores
- Ball Radar y TranSPORTmer (arriba).
- **Graph Imputer** (Omidshafiei et al., 2022): imputa jugadores fuera de plano en tele, con grafos.
- **Lección**: con un 10-20 % del balón visible, el equipo poseedor sube al 91-95 % en Ball Radar. Nosotros
  tenemos el balón el 59 % del tiempo, pero **en huecos largos** (mediana 10,7 s), un caso más difícil que el
  enmascarado aleatorio por frame del artículo.

## 4. Más datos de entrenamiento con posesión
| fuente | qué trae | acceso | encaje |
|---|---|---|---|
| **SkillCorner open data** | 10 partidos de la A-League: tracking de tele, eventos dinámicos y **fases de juego** | GitHub público ([skillcorner/opendata](https://github.com/SkillCorner/opendata)) | 4.ª fuente del banco; es de tele (jugadores fuera de plano), parecido a nuestros huecos |
| **PFF FC Mundial 2022** | 64 partidos de tracking de tele + eventos | Gratis con formulario ([blog de PFF](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset)); licencia por confirmar | ×6 los partidos de entrenamiento; hay que comprobar si trae el estado del balón y el poseedor |
| Metrica (3) y DFL (7) | ya en el banco | — | — |

**Antes de usarlos**: confirmar la licencia y que las bases del reto permiten datos externos para entrenar
(RULES dice que sí a datos externos; Q3 en `docs/00_reto.md`).

## 5. Adaptación al partido real
- **Autoentrenamiento con pseudoetiquetas**: estándar en adaptación de dominio. Para series temporales,
  TransPL (arXiv 2505.09955) usa matrices de transición de códigos discretos. Riesgo conocido: refuerza los
  errores seguros del modelo, como «sin balón = parón».
- **Normalización por partido**: ya la hacemos en las familias (z-score relativo al partido).

## 6. Visión (vídeo táctico)
| técnica | evidencia | encaje |
|---|---|---|
| **TrackNet V3**, WASB (arXiv 2311.05237) | Muy buenos con el balón rápido y sin oclusión; **fallan con el balón parado o en conducción**; falsos positivos con botas y líneas | Nuestro YOLO afinado falla igual (E068: balón pegado a un jugador en los parones) |
| **SAHI / mosaicos** | Estándar para objetos pequeños; ya usamos mosaicos de 640 | — |
| **SoccerNet Ball Action Spotting** (T-DEED, 1.º en 2024; TBAS-1 en 2025) | 12 acciones con el balón (pase, conducción...) y equipo; Team-mAP@1 s ≈ 60 en tele | Detectaría pases y robos en el vídeo, pero está entrenado en tele, no en plano táctico: dominio distinto |
| **Detección de jugadores + color de camiseta** (YOLO + clustering, p. ej. Sensors 2026) | Estándar | Completar los jugadores que faltan en el ATD (p5 = 14 visibles) |

## 7. Audio del broadcast
| técnica | evidencia | encaje |
|---|---|---|
| **PANNs / YAMNet** (AudioSet, 527 clases, versión por frame) | Traen *Whistle*, *Cheering*, *Crowd* y *Speech*; en una demo de `panns_inference`, *Whistle* sale como primera etiqueta | Silbato aprendido en vez de umbral espectral (E029 no encontró señal) |
| Detectores de silbato por banda espectral | 19,4 % de fallos en 12 h de audio, con muchos falsos positivos por la grada | Lo que ya probamos en E029 |
| **YAMNet + Whisper para resúmenes** (ICAART 2026) | Excitación del público y del comentarista como marcador de jugada | Tiros, córners y ocasiones → contexto de posesión ofensiva |
| **SoccerNet-Echoes** (Whisper sobre comentarios) | Falla con los nombres propios | Ya usamos la lista de nombres (E033: acuerdo de equipo 0,706) |

**Recordatorio**: Q6 (si se puede usar el audio del broadcast) **sigue sin enviarse**. Hasta tener respuesta,
el audio solo sirve para validar, no entra en la entrega.

## 8. Táctica: compacidad y formaciones
- **Compacidad por fase** (Castellano et al., AMISCO, 6 partidos de la Liga, 6.793 posesiones): anchura,
  profundidad y superficie **claramente mayores con balón**; la distancia del balón a la banda no cambia.
  Otro estudio con un partido: todas las medidas (centroide ponderado, stretch, superficie, área efectiva)
  cambian con y sin balón (p = 0,001). Contra rivales más flojos, los valores ofensivos son mayores.
  - **Para nosotros**: la señal es fuerte, pero **depende del equipo y del rival**. Por eso hay que medirla
    **relativa al propio equipo en el partido** (plantilla con y sin balón de cada equipo), no en bruto.
    Nuestras familias «bloques» y «estructura» usan z-score por partido, pero no comparan con la forma
    propia con y sin balón de cada equipo.
- **Formaciones** (Bialkowski et al., 2014-2016): roles por asignación húngara frame a frame (minimizar la
  entropía, parecido a k-means con asignación lineal). **Shaw y Glickman** (Barça Sports Analytics Summit):
  plantillas **con y sin balón** de cada equipo por clustering jerárquico con distancia de Wasserstein y
  clasificación bayesiana. **SoccerCPD** (Kim et al., KDD 2022, arXiv 2206.10926): change points de formación
  y de rol sobre matrices de adyacencia de roles.
  - **Para nosotros**: una feature «¿a qué se parece más la forma actual de cada equipo, a la suya con balón o
    a la suya sin balón?» es la traducción directa. Las plantillas se calculan sin GT, con los frames en que
    el modelo está seguro, como el umbral de toques. SoccerCPD trabaja a escala de minutos (cambios
    tácticos), no de segundos: sirve poco para fronteras de posesión.

## Fuentes
- PathCRF: https://arxiv.org/abs/2602.12080 · https://github.com/hyunsungkim-ds/pathcrf
- Ball Radar: https://arxiv.org/abs/2306.08206 · https://github.com/pientist/ballradar
- TranSPORTmer: https://arxiv.org/abs/2410.17785
- Multi-Modal Soccer Scene Analysis: https://arxiv.org/abs/2512.19528
- Vidal-Codina et al.: https://arxiv.org/abs/2202.00804
- Fases con GNN: https://arxiv.org/abs/2610.11571
- MS-TCN: https://arxiv.org/abs/1903.01945 · ASRF: https://arxiv.org/abs/2007.06866
- TransPL: https://arxiv.org/abs/2505.09955
- WASB: https://arxiv.org/abs/2311.05237 · TrackNet: https://arxiv.org/abs/1907.03698
- SoccerNet 2024/2025: https://arxiv.org/abs/2409.10587 · https://arxiv.org/abs/2508.19182
- Audio para resúmenes: https://arxiv.org/abs/2501.16100 · https://arxiv.org/abs/2609.17923
- SoccerCPD: https://arxiv.org/abs/2206.10926 · Bialkowski/roles: https://arxiv.org/abs/2502.03342
- Shaw y Glickman: https://www.sportperformanceanalysis.com/s/Dynamic-analysis-of-team-strategy-in-professional-football-By-Laurie-Shaw-And-Mark-Glickman.pdf
- Compacidad (Castellano et al.): https://ddd.uab.cat/pub/revpsidep/revpsidep_a2013v22n2/revpsidep_a2013v22n2p437.pdf
- SkillCorner: https://github.com/SkillCorner/opendata · PFF: https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset
