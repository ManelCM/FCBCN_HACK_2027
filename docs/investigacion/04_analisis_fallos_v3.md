# Análisis de fallos del modelo de la entrega (E070), banco v3 y reto, explicado

- **Fecha**: 2026-10-25 · **Autor**: Manel (con agente) · **Rama**: `analisis-fallos-v3`
- **Modelo analizado**: el de la entrega (E070): apilado + cinco familias, decodificación fija 1/20/10/10,
  `La MasIA.csv` sha `9602959c4ec05bd6`.
- **Experimentos**: E039-E046 (`docs/experimentos.md`). **Salidas en bruto**: `docs/investigacion/resultados/`
  (`fallos_v3_e070.*`, `real_vs_bench.*`, `atribucion_d_cortos.txt`, `balon_lento.txt`, `viabilidad_c*.txt`).
- **Predecesor**: `02_analisis_fallos.md` (22 oct, modelo viejo, solo Metrica). Este lo sustituye para el
  modelo actual.

## Resumen en un minuto
1. **En el banco v3 (10 partidos) el modelo saca 0,7206** con la decodificación de la entrega (0,7185 eligiendo la
   decodificación con los otros 9). Se pierden el 29 % de las posesiones reales a IoU 0,5.
2. **El margen está en ver los cambios de posesión, no en el Viterbi.**
   - Arreglar los cambios H↔A que no vemos vale +0,146: +0,097 los que ocurren **con el balón visible** y
     +0,054 los de los huecos.
   - Quitar los cambios H↔A que nos inventamos vale +0,090.
   - Afinar la decodificación da como mucho +0,001 (+0,011 en el caso optimista).
   - El filtro de duración mínima no da nada.
3. **En el reto hay un fallo que el banco no puede ver: D cortos sobrantes.**
   - Predecimos 0,68 tramos D de menos de 5 s por minuto. En la GT del banco hay 0,09 y en nuestra predicción
     del banco, 0,12: **unas seis veces más**.
   - **El 90 % aparecen con el balón visible**. Unos 50 parten una posesión en dos y unos 22 sustituyen a un
     cambio directo.
   - Por eso en el reto predecimos un 31 % menos de cambios directos por minuto (1,40 frente a 2,03 de la GT del
     banco) y las posesiones salen más cortas (mediana 6,0 s frente a 7,3 s).
4. **Causa encontrada.** La atribución con TreeSHAP señala `ball_speed` y `touch_rank`: en esos D cortos el balón
   del ATD va lento (mediana 1,2 m/s), y el modelo aprendió en el banco que balón lento es parón.
   - **El simulador del banco borra el balón lento en juego vivo**: el error correlado del balón (`ball_corr_noise`
     0,8 m, τ 75 frames) mete unos 4 m/s de velocidad aparente.
   - Tramos de balón lento ≥ 1 s por minuto de juego vivo: banco limpio 1,08; banco degradado 0,01; reto 0,55.
   - **Cuánto cuesta**: inyectados en el banco a la tasa del reto, los D cortos bajan la nota de 0,721 a
     **0,633 (−0,088)**. Es el fallo más caro que hemos encontrado y **la primera mejora del plan**.
5. **Puertas de las líneas alternativas**:
   - **audio** (PANNs, preentrenado en AudioSet): sin señal de silbato (AUC 0,51-0,54). Cerrado.
   - **YOLO como evidencia de equipo**: fiable (92,4 % de acuerdo, igual que el balón del ATD), pero con poco
     recorrido (3,9 min de juego vivo en hueco).
   - **compacidad**: ya la explotamos. En nivel separa bien (AUC 0,87), pero **se invierte en los 3 s tras un
     robo** (0,36), y entonces solo informa el cambio (0,82).
   - **Ball Radar y PathCRF**: definidas como experimentos; no se han corrido.

## 1. Qué se ha medido y cómo
- **Banco v3**: 3 Metrica + 7 DFL degradados como el ATD (ADR-0014), GT controlada (T = 1,5 s en Metrica y 2 s
  en DFL), probabilidades fuera de muestra del modelo de la entrega (OOF guardado de E035/E036,
  `external_data/oof_l3_notags_v3_ftouch+...+restart_stack_dyn_ctl1.5_dfl2_trainraw.npz`). La reproducción da
  0,7185 dejando un partido fuera, igual que E035 (E039).
- **Herramienta**: `pipeline/failure_analysis.py --bench v3` (antes solo cubría Metrica). Novedades:
  - eventos de DFL con kloppy, traducidos al esquema de Metrica (`io/external.dfl_events`). El desfase por parte se
    estima contra el balón del tracking: 1-46 frames. La distancia mediana evento-balón queda en 3,5-5,5 m, **por
    encima de la puerta de 2 m**: los eventos de DFL se apuntan a mano y van desfasados. Por eso en DFL el contexto
    de eventos usa una ventana de ±1 s y hay que leerlo con cautela. La orientación sí es correcta: espejado, la
    distancia sería de 57 m. El equipo del evento coincide con el de la GT en el 88-91 %;
  - tabla por partido;
  - descomposición del margen (§3).
- **Reto** (sin GT, solo agregados, `scripts/run_aggregate.sh`):
  - `pipeline/real_vs_bench.py`: forma de lo que predecimos frente a la GT del banco, la predicción del banco y la
    GT pública;
  - `pipeline/short_dead_report.py`: atribución con TreeSHAP;
  - `pipeline/frozen_ball.py`: velocidad del balón;
  - `public_window`: ventana pública, autorizada.
- **Qué es un oráculo**: sustituir nuestra predicción por la GT solo en ciertos frames y recalcular. Es el techo
  de arreglar eso. **No se suman** entre sí.

## 2. La nota y cómo se reparte
| umbral de IoU | precisión | recall | F1 |
|---|---|---|---|
| 0,3 | 0,854 | 0,798 | 0,825 |
| 0,5 | 0,757 | 0,707 | 0,731 |
| 0,7 | 0,615 | 0,574 | 0,594 |

- 2.694 posesiones predichas frente a 2.885 reales: seguimos **infrasegmentando** un 7 %.
- De 0,3 a 0,7 caemos 0,23. Las fronteras importan, pero sobre todo porque **faltan o sobran cambios enteros**
  (§3), no porque los que vemos estén mal colocados.

**Por qué se pierde cada posesión real** (846 perdidas a IoU 0,5, el 29,3 %):
| causa | n | % de las perdidas | duración mediana |
|---|---|---|---|
| la damos al equipo contrario | 342 | 40,4 % | 3,3 s |
| fusionada: no vemos una posesión corta del rival | 192 | 22,7 % | 4,1 s |
| tapada por D | 121 | 14,3 % | 4,2 s |
| fusionada: posesión rival y parón | 90 | 10,6 % | 4,7 s |
| partida en trozos | 45 | 5,3 % | 16,0 s |
| fronteras desplazadas | 40 | 4,7 % | 4,3 s |
| fusionada: parón no detectado | 16 | 1,9 % | 3,4 s |

Además sobran 655 posesiones predichas: 337 solapan una real pero mal ajustadas, 202 son de un equipo que no
tiene el balón y 116 son trozos cortos.

**Dónde**:
- **Por duración**: encontramos el 43 % de las posesiones de menos de 3 s, el 67 % de las de 3-10 s y el 88 % de
  las de más de 10 s.
- **Por pases propios**: con 0 pases encontramos el 52 %; con 6 o más, el 96 %. **Lo que fallamos son posesiones
  cortas, sin pases**: un equipo recupera, aguanta 2-5 s y la pierde.
- **Por huecos**: si más de la mitad de la posesión cae en un hueco largo, encontramos el 51 %; sin hueco, el 76 %.
- **Por cómo acaban**: tras un pase interceptado, 77 %; tras una pérdida sin evento, 55 %.
- **Por cómo empiezan**: córner 41 %, saque de centro 91 %.

**Fronteras** (transiciones reales con una predicha del mismo tipo cerca):
| tipo | n | a ≤ 0,5 s | ≤ 1 s | ≤ 2 s | falsas |
|---|---|---|---|---|---|
| cambio H↔A | 1.928 | 37,7 % | 51,3 % | 63,0 % | 467 |
| paso a D | 951 | 66,9 % | 71,3 % | 74,7 % | 296 |
| reanudación | 940 | 59,5 % | 73,8 % | 83,5 % | 208 |

**Frame a frame**:
- Con balón visible acertamos el 0,922.
- En huecos largos, el 0,892. Pero ahí **el juego vivo sale D el 22-25 %** (GT H → D 0,25; GT A → D 0,22),
  mientras que el D se acierta el 0,98.
- Cerca de un duelo acertamos el 0,778; en vuelo, el 0,947.
- Pases perdidos: los cortos interceptados se aciertan el 0,65 en vuelo y en el segundo siguiente.

## 3. Descomposición del margen (oráculos, base 0,7206)
| escenario | spellF1 | ganancia | lectura |
|---|---|---|---|
| equipo perfecto en los frames vivos | 0,980 | +0,259 | techo absoluto: casi todo es «quién» |
| fronteras de la GT + etiqueta del modelo | 0,844 | **+0,123** | error de «quién» (etiquetas de tramos bien cortados) |
| fronteras del modelo + etiqueta de la GT | 0,804 | **+0,083** | error de «dónde» (cortes) |
| **todos los cambios H↔A no vistos, arreglados** | 0,867 | **+0,146** | 713 de 1.928 cambios reales no tienen un cambio nuestro a ≤ 2 s |
| … solo los no vistos **con balón visible** (480) | 0,818 | **+0,097** | **la mayor parte: no es un problema de huecos** |
| … solo los no vistos en hueco (233) | 0,775 | +0,054 | — |
| **cambios H↔A sobrantes, quitados** (467) | 0,811 | **+0,090** | nos inventamos uno de cada cuatro cambios |
| huecos largos perfectos | 0,819 | +0,098 | balón ideal en los huecos (techo del relleno) |
| la mitad de los huecos perfectos | 0,768 | +0,047 | relleno realista |
| posesiones < 3 s perfectas | 0,792 | +0,072 | — |
| causa «equipo contrario» resuelta | 0,857 | +0,136 | — |
| causa «posesión rival corta no vista» resuelta | 0,801 | +0,081 | — |
| fronteras vistas ajustadas a ≤ 0,5 / 1 / 2 s | 0,727 / 0,736 / 0,753 | +0,006 / +0,015 / +0,032 | **colocar mejor lo que ya vemos vale poco** |
| D perfecto | 0,735 | +0,014 | el D importa poco en el banco |
| tramos D predichos sin D real, arreglados | 0,740 | +0,019 | — |
| mejor rejilla común del Viterbi | 0,722 | +0,001 | **la decodificación está agotada** |
| mejor rejilla por partido (optimista) | 0,732 | +0,011 | — |
| filtro de duración mínima 0,5 / 1 / 2 s | 0,721 / 0,721 / 0,717 | 0 / +0,001 / −0,004 | un HSMM sencillo no aporta |
| regla «D < k s entre vivos → tramo anterior» | 0,721 … 0,713 | 0 … −0,008 | en el banco no hay D cortos que quitar (§6) |
| **D cortos del reto inyectados** (tasa del reto / la mitad) | 0,633 / 0,674 | **−0,088 / −0,047** | lo que nos cuesta el defecto del reto (§6.3) |
| argmax sin Viterbi | 0,359 | −0,362 | las probabilidades por frame solas no sirven |

**Lectura**:
- Las probabilidades son buenas (macroF1 0,90), y el Viterbi y las fronteras ya sacan casi todo lo que pueden.
- Lo que falta es **evidencia para ver los robos cortos**, que suelen ocurrir con el balón visible y sin pases, y
  **para no inventar cambios**.
- Atacarlo pide una señal nueva, no afinar lo que hay.

## 4. Por partido (y J03WR9)
| partido | spellF1 | F1@0,3 | F1@0,7 | posesiones pred./GT | hueco largo | D | acierto (visible / hueco) |
|---|---|---|---|---|---|---|---|
| Metrica 1 | 0,745 | 0,833 | 0,636 | 259/298 | 36,7 % | 33,4 % | 0,901 / 0,810 |
| Metrica 2 | 0,721 | 0,827 | 0,594 | 248/260 | 42,2 % | 31,7 % | 0,896 / 0,784 |
| Metrica 3 | 0,750 | 0,832 | 0,657 | 262/310 | 35,4 % | 25,3 % | 0,932 / 0,783 |
| J03WMX | 0,711 | 0,829 | 0,573 | 281/298 | 39,9 % | 41,5 % | 0,938 / 0,919 |
| J03WN1 | 0,680 | 0,799 | 0,526 | 286/315 | 43,9 % | 51,7 % | 0,924 / 0,916 |
| J03WPY | 0,752 | 0,846 | 0,651 | 272/272 | 42,3 % | 44,3 % | 0,928 / 0,958 |
| J03WOH | 0,748 | 0,854 | 0,601 | 257/279 | 47,8 % | 47,6 % | 0,936 / 0,948 |
| J03WQQ | 0,725 | 0,829 | 0,598 | 295/284 | 40,5 % | 40,7 % | 0,932 / 0,918 |
| J03WOY | 0,741 | 0,842 | 0,599 | 282/302 | 46,2 % | 46,5 % | 0,932 / 0,926 |
| **J03WR9** | **0,631** | 0,755 | **0,501** | 252/267 | 40,7 % | 41,9 % | 0,902 / 0,922 |

- **J03WR9 no falla por los huecos ni por el D**: su acierto en huecos (0,922) es normal. Falla en las
  **fronteras**: F1@0,7 = 0,50, el peor, y ya a IoU 0,3 está 0,07 por debajo del resto. Su acierto con balón
  visible es el más bajo de DFL (0,902).
- Encaja con E060: sin la familia «balón», J03WR9 sube a 0,679. La sospecha es que «balón quieto» y «vuelo
  largo» aprenden artefactos del simulador, y el §6 los confirma: **el balón lento del simulador es irreal**.
- En Metrica el acierto en huecos (0,78-0,81) es mucho peor que en DFL (0,92-0,96). Su GT sale de eventos, no del
  tracking, y el D dentro de sus huecos se define de otra manera.

## 5. Reto frente a banco: lo que predecimos (E041)
Agregados de lo que predice el modelo en el partido real (105,9 min puntuados) frente a la GT del banco (lo que
«debería» salir si el reto se pareciera al banco), a nuestra predicción en el banco (el sesgo propio del modelo)
y a la GT pública.

| medida | reto (pred.) | GT pública 5 min | pública (pred.) | banco (GT) | banco (pred.) |
|---|---|---|---|---|---|
| posesiones por minuto | 3,17 | 3,40 | 3,60 | 3,03 | 2,83 |
| duración p50 / media (s) | 6,0 / 10,4 | 7,6 / 14,2 | 10,5 / 14,4 | 7,3 / 11,8 | 7,9 / 12,1 |
| % de posesiones < 3 s | **28,0 %** | 17,6 % | 22,2 % | 17,5 % | 17,4 % |
| % de D | 44,8 % | 19,6 % | 13,5 % | 40,4 % | 43,0 % |
| tramos D por minuto | **1,78** | 0,80 | 2,40 | 1,00 | 1,06 |
| duración mediana de un tramo D (s) | **7,9** | 16,4 | 2,5 | 18,3 | 18,7 |
| **tramos D < 5 s por minuto** | **0,68** | — | — | 0,09 | 0,12 |
| … que empiezan con el balón visible | **90 %** | — | — | 62 % | 48 % |
| … entre dos tramos del mismo equipo, por minuto | **0,47** | — | — | 0,035 | 0,048 |
| … entre equipos distintos, por minuto | 0,21 | — | — | 0,049 | 0,064 |
| cambios H↔A directos por minuto | **1,40** | 2,40 | 1,00 | 2,03 | 1,77 |
| … con balón visible / en hueco largo | 2,24 / **0,16** | — | — | 2,78 / 0,98 | 2,61 / 0,59 |
| % de D dentro de los huecos largos | 91 % | — | — | 71 % | 77 % |
| % de D con balón visible | 13 % | — | — | 18 % | 18 % |
| % de frames con pmax < 0,7 (visible / hueco) | 9,1 % / 6,5 % | — | — | — | 7,0 % / 6,9 % |

**Lectura**:
- En nuestra predicción, el reto tiene el mismo número de posesiones que el banco, pero con otra composición:
  **cambia «H → A» por «H → D corto → A»** y parte posesiones con D cortos.
- En los huecos casi no hay cambios directos (0,16 frente a 0,59 de nuestra predicción en el banco): el modelo
  convierte casi todo hueco en D (91 % frente a 77 %).
- El % total de D (44,8 %) es verosímil para un partido (≈ 58 min de juego efectivo en 106). Lo anómalo es
  **cómo** se reparte.
- La confianza del modelo es la misma en el reto que en el banco: **se equivoca seguro**, así que una regla por
  confianza no lo detecta.

## 6. La causa de los D cortos: el balón lento (E042-E043)
### 6.1 Atribución (TreeSHAP sobre la capa 1, 1.500 frames por grupo)
Contribución media al logit de D en tres grupos de frames del reto:
- dentro de los D cortos con balón visible (S): **+2,65**;
- juego vivo con balón visible (V): −4,14;
- D largos (L): +4,57.

Features que más empujan hacia D en S frente a V:
| feature | S − V | z en S / V / L (reto) | banco: z vivo / D |
|---|---|---|---|
| `ball_speed` | **+2,95** | −0,66 / −0,07 / −0,66 | +0,31 / −1,03 |
| `touch_rank` | **+1,22** | −1,20 / −0,02 / −1,23 | +0,35 / −1,19 |
| `speed_all_m125`, `speed_all_m50` (ritmo colectivo) | +0,30 | +0,02 / +0,74 / −0,54 | +0,67 / −0,67 |
| `ball_still_m25` | +0,26 | +0,59 / −0,22 / +1,22 | −0,45 / +1,17 |
| `recv05_h`, `recv10_h` (receptor) | +0,20 | **+3,75 / +13,1** en S | ≈ 0 |

- Por familias: balón +3,45, toques +1,35, velocidad colectiva +0,77.
- **En los D cortos el balón está tan lento y con tan pocos toques como en un parón real, pero los jugadores se
  siguen moviendo a ritmo intermedio**: es juego vivo con el balón en los pies.
- `recv05_h`/`recv10_h` toman valores extremos (z +4 a +13) en esos frames: posible inestabilidad numérica del
  receptor con balón lento. Hay que revisarlo.

### 6.2 El simulador borra el balón lento del juego vivo
| | velocidad p25 / p50 (m/s) | % de frames lentos (< 1,25 m/s) | tramos lentos ≥ 1 s por minuto |
|---|---|---|---|
| reto, vivo predicho | 3,3 / 6,0 | 5,6 % | **0,55** |
| reto, D corto predicho | 0,6 / 1,2 | 52 % | 11,4 |
| reto, D largo predicho | 0,2 / 0,9 | 63 % | 7,6 |
| banco **limpio**, vivo (GT) | — | 8,8 % | **1,08** |
| banco **degradado**, vivo (GT) | 4,4 / 7,1 | 3,3 % | **0,01** |
| banco degradado, D (GT) | 0,6 / 1,0 | 69 % | 0,73 |

- En el tracking limpio hay balón lento en juego vivo: un central que conduce o un jugador que protege el balón.
- **Tras degradar desaparece**, aunque el 78 % de esos frames siguen viendo el balón.
- La culpa es de `_ball_signal` en `io/degrade.py`: el error correlado (`ball_corr_noise` 0,8 m, `ball_corr_tau`
  75) es un proceso de Ornstein-Uhlenbeck **sin suavizar**, con unos 0,13 m de salto por frame y por eje, que
  equivale a unos 4 m/s de velocidad aparente.
- Se calibró en E013 contra el residuo y las patadas del ATD, pero no contra la velocidad del balón lento.
- Resultado: **en el banco, un balón visible y lento solo aparece en los parones**, y el modelo aprende «lento =
  D». En el reto, el balón lento del juego vivo dispara D cortos.
- **Por qué el banco no lo mide**: su degradación no genera la situación, así que los oráculos y reglas de D
  cortos del §3 salen a cero. **El protocolo de selección (ADR-0008) no puede ver este fallo con el simulador
  actual**: hay que arreglar el simulador primero.

### 6.3 Cuánto cuesta
Inyectamos en las predicciones del banco los D cortos de más que tiene el reto, restando lo que el banco ya tiene
de serie: 0,42 por minuto partiendo posesiones y 0,14 en los cambios, en tramos de 1-4,5 s, media de 5 semillas
(`failure_analysis`, §9):

| D cortos de más | spellF1 | diferencia |
|---|---|---|
| ninguno (base) | 0,7206 | — |
| la mitad de la tasa del reto | 0,6741 | **−0,047** |
| la tasa del reto | 0,6330 | **−0,088** |

- **Es el fallo más caro que hemos encontrado**: cuesta casi lo mismo que todos los cambios sobrantes del banco
  juntos (+0,090).
- Explica buena parte de la distancia entre el banco (0,72) y lo que esperamos del reto: la ventana pública da
  0,651, con mucho ruido.
- Si M1 lo elimina, el reto debería recuperar del orden de **+0,05 a +0,09** sin que el banco cambie.

## 7. Puertas de las líneas alternativas (fase C del plan)
| línea | qué se midió | resultado | veredicto |
|---|---|---|---|
| **C1 YOLO** como evidencia de equipo en los huecos (`vision/yolo_report.py`, E044) | Cobertura del balón del vídeo según lo predicho; acuerdo «equipo más cercano al balón del vídeo» con el equipo predicho | Cobertura 55 % donde predecimos D y 63 % donde predecimos vivo. Acuerdo **0,924** con pmax ≥ 0,9 (balón del ATD: 0,927); 0,671 con pmax < 0,7. Solo 3,9 min de juego vivo predicho en hueco y 2 candidatos a robo no visto | Señal **fiable** pero con **poco recorrido** mientras el modelo convierta los huecos en D. Vuelve a tener sentido si el arreglo del §6 devuelve juego vivo a los huecos |
| **C2 Audio** con PANNs (AudioSet, `audio/audioset_gate.py`, E045) | Probabilidad de *Whistle* en ±0,5 s del inicio de D largos y cortos y de las reanudaciones, frente al azar; grada y habla | AUC 0,510 / 0,544 / 0,520; grada 0,003 frente a 0,004; habla 0,56 frente a 0,55 | **Sin señal**, como en E029. El audio queda cerrado salvo el comentario (E033: equipo 0,706) |
| **C4 Compacidad y forma** (`pipeline/gate_shape.py`, E046) | AUC de «posee el local» de cada señal de forma (local − visitante) en juego vivo, en huecos y en los 0-3 s tras un robo | Nivel de anchura, superficie y stretch: **0,85-0,87** en vivo y 0,81 en hueco, pero **0,36-0,37 tras un robo** (invertido: el que roba aún está cerrado). Sus cambios en 3 s: 0,58 en vivo pero **0,78-0,82 tras un robo**. Modelo: 0,965 / 0,919 / 0,860 | La literatura se cumple (con balón, más anchos), y **ya lo explotamos**: el modelo supera a cualquier señal suelta. Plantillas de formación con y sin balón: poco recorrido esperado. Lo útil: tras un robo el nivel engaña y manda el cambio |
| **C3 Ball Radar / TranSPORTmer** | No ejecutado (2-3 días) | — | Definido como experimento en `docs/05_plan_mejora.md` |
| **C5 Modelos** (PathCRF, MS-TCN, ASRF) | No ejecutado como entrenamiento; la descomposición del §3 da su techo | Decodificación +0,001, fronteras ≤ 1 s +0,015: MS-TCN, ASRF y HSMM atacan poco margen. PathCRF ataca «quién» y los cambios (+0,10-0,15 de techo) | PathCRF / poseedor individual, en el plan |

## 8. Ventana pública con E070
- spellF1 0,651; 30 segmentos predichos frente a 21 reales.
- **Los 11 cambios sobrantes pasan todos por D**: casi todos son D de 0,3-4 s.
- Se siguen sin ver las microposesiones de < 1,5 s (2:46 y 3:03) y la H de 3,5 s de 4:04.
- Detalle en `03_errores_ventana_publica.md`.

## 9. Comparación con el análisis del 22 oct (modelo viejo, solo Metrica)
| | 22 oct (apilado v3, Metrica) | 25 oct (E070, banco v3) |
|---|---|---|
| spellF1 | 0,628 | 0,7206 (en Metrica: 0,721-0,750) |
| cambios H↔A con frontera a ≤ 2 s | 49 % | 63 % |
| acierto en huecos largos | 0,78 | 0,89 (Metrica 0,78-0,81) |
| juego vivo → D en huecos | 22 % | 22-25 % (sin mejora) |
| causa principal | equipo contrario | equipo contrario (40 %) |
| techo de los huecos perfectos | +0,158 | +0,098 |

Las familias han reducido el problema de los huecos y han mejorado las fronteras, pero el juego vivo que sale D
dentro de los huecos sigue igual. Ahora sabemos que en el reto es aún peor (§5).

## 10. Límites
- Los oráculos no se suman y son techos, no ganancias.
- El contexto de eventos de DFL va con una ventana de ±1 s por el desfase de sus eventos manuales.
- En el reto **no hay GT**: «vivo» y «D» son lo que predecimos. Las tasas del reto comparan nuestra predicción con
  la GT del banco y la diferencia puede mezclar dos cosas: el error del modelo y diferencias reales entre partidos.
  La señal de los D cortos (×6, el 90 % con balón visible) es demasiado grande para ser estilo de juego, y la
  atribución y el balón lento la explican.
- TreeSHAP con equipos intercambiados reparte la contribución entre las parejas `_h`/`_a`: las cifras por familia
  son más robustas que las de cada feature.
