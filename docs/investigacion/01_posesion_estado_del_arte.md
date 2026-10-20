# 01 — Posesión por frame: análisis del problema, estado del arte y plan técnico

_Autor: Manel (con Claude) · 2026-10-20 · Basado en las bases oficiales (ver `docs/00_reto.md`). Lo que no se ha comprobado con datos está marcado como **[VERIFICAR]**._

---

## 1. Qué premia realmente la métrica (y qué no)

`spellF1` evalúa la **segmentación** del partido en posesiones, no la etiqueta de cada frame. De la definición (`possession_metrics.py`) salen consecuencias directas:

1. **El número de posesiones importa tanto como acertar el equipo.** Cada "parpadeo" H→A→H crea dos posesiones falsas y además rompe una verdadera, que deja de emparejarse (su IoU cae). El baseline sin suavizar tiene macroF1 0,65 pero spellF1 0,05.
2. **La precisión temporal de las fronteras es crítica.** 3 frames (0,12 s) de retraso apenas cuestan (0,98), pero **1 s cuesta 35 puntos** (0,65). La posesión media dura unos 14 s; en posesiones cortas, un error de 1-2 s hace que no superen IoU 0,5-0,7.
3. **El balón muerto parte posesiones.** `H D H` son dos posesiones, así que **detectar bien los `D`** (inicio y fin) es tan importante como detectar los cambios H↔A. Una falta no detectada fusiona dos posesiones y un `D` falso en medio de una jugada parte una en dos.
4. **`X` no tiene ninguna ventaja estratégica.** En spellF1, `X` = repetir la última etiqueta; en macroF1, `X` es siempre fallo. Por tanto, **escribir explícitamente la etiqueta anterior domina a `X`**: mismo spellF1 y macroF1 igual o mejor. Decisión: nunca escribir `X` en frames evaluados; solo fuera de ellos.
5. **La GT es mecánica**: sale del *tracking corregido* de Metrica (flags de balón vivo/muerto + equipo poseedor), sin suavizar. Hay que imitar **cómo marca Metrica la posesión en su tracking corregido**, no lo que diría un comentarista. Probablemente el cambio se marca en el instante del primer contacto controlado del nuevo equipo **[VERIFICAR con la GT pública: mirar en qué frame exacto cambia la etiqueta respecto al toque]**.
6. Números orientativos: 17 posesiones en 5 min → **~350-400 posesiones en el partido** (estimación nuestra; la cifra real no se publica a propósito). El ruido de spellF1 es ±0,04.

## 2. Dificultades del dato
- **Balón ausente en el 41 % de frames evaluados**, concentrado en duelos, áreas y balones aéreos.
- Tracking **sin identidades** (46 slots para 22 jugadores, que se fragmentan e intercambian) → solo features **colectivas** o por proximidad, nunca "el jugador 7".
- Smart Tagging **automático**: son fases, no posesiones, y los saques de banda tienen `Team = N/A`.
- **Una sola GT pública de 5 min** → no podemos elegir modelos con ella (el repo lo demuestra con su propio baseline).

## 3. Literatura y lo que aporta

| Trabajo | Idea | Resultado | Para nosotros |
|---|---|---|---|
| Link & Hoernig et al. (2016), *Detection of individual ball possession in soccer* | Física del balón + ML para detectar control | P/R 80-92 % con datos limpios | Definir el **control** por cinemática (toque), no solo por proximidad |
| Vidal-Codina et al. (2022, MIT/FIFA), *Automatic event detection in football using tracking data* | Árbol de decisión: posesión frame a frame + eventos a partir de los cambios de posesión | >90 % en la mayoría de eventos | Reglas base; implementado en DataBallPy (radio 1,5 m, velocidad del balón 5 m/s, ángulo 10°) |
| Borghesi et al. (2023), *TCNs to estimate ball possession* | Red temporal sobre posiciones; posesión + vivo/muerto | 86,2 % posesión, 89,2 % vivo/muerto | Lo **secuencial** gana a las reglas; hay un modelo explícito de **vivo/muerto** |
| arXiv 2306.08206 (Set Transformers) | Representación invariante al orden de los jugadores y reconstrucción del balón | — | Encaja con slots sin identidad; **inferir el balón cuando falta** |
| arXiv 2501.10299 (transporte óptimo) | Equipo en posesión **solo con las posiciones de los jugadores** | 81,8 % | La forma de los equipos aguanta cuando falta el balón |

## 4. Idea diferencial: datos externos con la misma clase de etiqueta

El gran problema es que **no tenemos con qué validar ni entrenar** (5 min de GT). Pero las reglas permiten **datos externos**, y existen datasets abiertos de tracking **corregido** que traen, por frame, **estado del balón (vivo/muerto) y equipo poseedor**. Es exactamente cómo se construye la GT del reto:

- **DFL / Sportec open data** (Bassek et al., 2024, *Scientific Data*): 7 partidos de Bundesliga y 2. Bundesliga 2022/23, a 25 fps, con tracking y eventos. kloppy los carga (`sportec`) y expone `ball_state` (alive/dead) y, según el formato DFL, el equipo poseedor por frame **[VERIFICAR campo `ball_owning_team`]**.
- **Metrica sample-data** (3 partidos anonimizados, tracking + eventos). La posesión se puede derivar de los eventos y Metrica es el mismo proveedor que la GT **[VERIFICAR qué flags trae el EPTS]**.
- Otros posibles: PFF FC World Cup 2022 (tracking + eventos) y SkillCorner open data (tracking de broadcast **con balón ausente**, muy parecido a nuestro ATD) **[VERIFICAR licencias y campos]**.

Plan:
1. **Degradar** el tracking limpio para que se parezca al ATD: quitar el balón aproximadamente un 41 % del tiempo y **más cerca de los jugadores y en el aire**, fragmentar tracks, quitar identidades y añadir ruido de posición. Los parámetros se calibran midiendo nuestro ATD (huecos, longitud de tracks, ruido); eso son estadísticas que medimos en local, no se exportan.
2. **Entrenar** el modelo (fase L3/L4) sobre los datos externos degradados.
3. **Validar** con los partidos externos que dejamos fuera (**cientos de posesiones** frente a 17) usando **la misma función** `spell_scores` del scorer oficial. Así se resuelve "no hay leaderboard".
4. La GT pública de 5 min solo se usa como **prueba de humo** (alineación, formato, sesgo de frontera).

Riesgo: diferencias de definición entre proveedores (DFL frente a Metrica) → hay que medir el sesgo de frontera en los 5 min públicos y ajustarlo como un desfase global, no por partido.

## 5. Arquitectura propuesta

```
ATD (tactical) ─► A. Limpieza ─► B. Balón vivo/muerto ─► C. Detección de toques/control ─► D. Decodificación secuencial ─► CSV
Smart Tagging ─────────────────────────┘ (reanudaciones)                                ▲
Features colectivas (forma, velocidades, presión) ──────────────────────────────────────┘
```

- **A. Limpieza del tracking**: quitar detecciones de balón imposibles (saltos > ~40 m/s), interpolar huecos cortos del balón con física (lineal o parabólica), unir tracklets.
- **B. Vivo/muerto**: balón fuera de los límites; firmas de parada (velocidad media de los jugadores baja, agrupamientos tipo córner/falta, jugador en la banda para un saque, configuración de saque de centro); **anclas de reanudación** de Smart Tagging (`SET PIECES`: el inicio del set piece marca el fin del `D`).
- **C. Toques / control**: el balón cerca de un jugador (radio < ~1,5 m) con un cambio brusco de velocidad o dirección → candidato a toque de ese equipo. "Control" = toques consecutivos del mismo equipo o permanencia con el balón (proxies de CHALLENGE.md: unos segundos con el balón o más de 2 pases seguidos).
- **D. Decodificación**: **HMM / semi-Markov (HSMM)** sobre los estados {H, A, D} con:
  - emisiones = probabilidades por frame de un clasificador (LightGBM) con features de proximidad, cinemática y forma colectiva (cuál de los dos equipos ataca: anchura, profundidad, altura de la línea defensiva, velocidad media hacia la portería rival);
  - transiciones penalizadas y **prior de duración** (las posesiones de < 1 s son raras) → Viterbi = segmentación óptima. Ataca directamente los problemas 1-3 de la sección 1.
  - Alternativa o mejora: modelo secuencial (TCN / BiGRU / Transformer) sobre ventanas, seguido de la misma decodificación.
- **Ajuste fino de fronteras**: cuando la decodificación dice "aquí cambia", buscar en ±1 s el frame del toque más probable (C) y mover ahí la frontera.

## 6. Escalera de entregas

| Nivel | Qué | spellF1 esperado | Para qué |
|---|---|---|---|
| L0 | Reproducir el baseline oficial | 0,24 | Validar el pipeline y el formato |
| L1 | Reglas: histéresis ("la posesión no cambia hasta que el rival controla") + `D` por balón fuera y set pieces | ? | **Primer fichero válido**, que guardamos siempre como reserva |
| L2 | B + C + HMM con emisiones heurísticas | ? | |
| L3 | LightGBM entrenado con datos externos degradados + HSMM | ? | Candidato principal |
| L4 | Modelo secuencial y ensemble | ? | Si L3 se estanca |

Regla: **siempre hay un fichero validado listo para subir** (y el Form se puede editar hasta el deadline). Lo subimos pronto y lo vamos actualizando.

## 7. Validación
- **Principal**: spellF1 sobre los partidos externos que dejamos fuera (sección 4), con *bootstrap*.
- **Secundaria**: diagnósticos del scorer en la ventana pública (desfase de las transiciones con tau 12 frente a 25 frames, número de posesiones, fragmentación).
- **Prohibido**: elegir hiperparámetros por el spellF1 de los 5 min públicos.
- Siempre se mira el **número de posesiones predichas** frente al esperado (~70 por cada 10 min, como sanity check).

## 8. Herramientas
`kloppy==3.18.0` (la versión fijada por la organización para el EPTS), numpy y pandas, LightGBM, `hmmlearn` o un Viterbi propio (son 30 líneas y lo controlamos mejor), mplsoccer o matplotlib para visualizar, y el scorer oficial como librería.

## Fuentes
- Repo oficial: https://github.com/metrica-sports/more-than-a-hack-2027
- Hoernig, Link et al.: https://portal.fis.tum.de/en/publications/detection-of-individual-ball-possession-in-soccer/
- Vidal-Codina et al.: https://arxiv.org/abs/2202.00804
- Borghesi et al. (TCN): https://iris.polito.it/handle/11583/2976470
- Set Transformers: https://arxiv.org/abs/2306.08206
- Transporte óptimo: https://arxiv.org/pdf/2501.10299
- DataBallPy (posesión, datos abiertos DFL): https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html · https://databallpy.readthedocs.io/en/latest/api_docs/dataparsers_api.html
- DFL open data (Bassek et al.): https://www.dshs-koeln.de/en/institut-fuer-trainingswissenschaft-und-sportinformatik/detail/meldung/official-game-data-in-the-journal-scientific-data · https://pypi.org/project/kloppy/
- Metrica sample data: https://github.com/metrica-sports/sample-data
