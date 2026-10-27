# Plan de mejora específico hacia ≥ 0,75 en el reto (25 oct → congelación del 14 nov)

> Sale de tres trabajos del 25 oct, en la rama `analisis-fallos-v3`:
> - el análisis de fallos (`docs/investigacion/04_analisis_fallos_v3.md`, E039-E046);
> - el estado del arte ampliado (`docs/investigacion/05_estado_del_arte_v2.md`);
> - las puertas de viabilidad de YOLO, audio y compacidad.
>
> Sustituye a la tabla de prioridades de `docs/03_plan_ganar.md` cuando el equipo lo apruebe. Cada mejora
> se acepta con el protocolo de ADR-0008: ≥ +0,01 y mejor en ≥ 7 de 10 partidos. M1 añade además un
> guardarraíl de forma en el reto (ADR nuevo).

> **26 oct: plan ampliado («plan completo»).** Cubre todo lo que se pidió los días 25 y 26, sumado y no
> sustituido:
> - todos los fallos;
> - internet «por todos lados»;
> - más modelos;
> - YOLO y audio para **todo** lo que ayude, no solo la posesión;
> - Ball Radar y TranSPORTmer;
> - compacidad y formaciones;
> - pensar como entrenador y analista.
>
> Está organizado en seis frentes (T0-T6) y cada línea lleva fecha, entregable y puerta. Las secciones M1-M6 de
> abajo siguen valiendo como detalle del frente T0. **Esta tabla se actualiza en cada sesión: nada sin
> estado.**

## Estado de la ejecución (se actualiza en cada sesión)
| frente | línea | fecha prevista | estado | resultado / entregable |
|---|---|---|---|---|
| T0 | M1.1 objetivos del balón del reto | 26 oct | ✅ | reproducen E013; balón lento en vivo ≥ 0,55/min (ventana pública con GT: 1,86) |
| T0 | M1.2-1.3 simulador v3.1 (E047, ADR-0017) | 26 oct | ✅ | pérdida 160 → 20,5; balón lento en vivo 0,006 → 0,96 tramos/min |
| T0 | E048: E070 evaluado en v3.1 | 26 oct | ✅ | 0,7206 → 0,7010 con la decodificación de la entrega; D cortos 0,119 → 0,186/min: el banco ya ve el defecto |
| T0 | E049: reentrenar en v3.1 | 26 oct | ✅ | 0,7254 frente a 0,7059 (10 de 10); con la decodificación de la entrega 0,7229 frente a 0,7006 y D cortos 0,180 → 0,107/min (GT 0,084) |
| T0 | E053: curva de aprendizaje (5 frente a 9 partidos) | 27 oct | ⏳ | — |
| T0 | E054: + 20 partidos de SkillCorner (GT completa H/A/D) | 27 oct | ⏳ | caché de los 20 partidos lista |
| T0 | M1.6 artefacto + guardarraíl (ADR-0018, umbrales pre-registrados) | 27 oct | ❌ E049c: no pasa (3 de 4); **guardarraíl relativo (ADR-0020): la entrega pasa a E075 (E075d); con la regla de D cortos de 3 s (ADR-0021, E090c), D cortos 0,198/min** | D cortos en el reto 0,680 → 0,529/min (umbral 0,25); tramos D 1,78 → 1,61; cambios 1,40 → 1,48 |
| T1 | búsqueda ampliada → `investigacion/06_estado_del_arte_v3.md` | 26-27 oct | ✅ 26 oct | Lang 2022 (estado sin balón), Link 2017 (toques ≥ 4 m/s²), Bauer y Anzer, pressing intensity, SigLIP/color, sonido del golpeo |
| T2 | F1 roles y formación (Bialkowski, Shaw y Glickman) (E073) | 26 oct | ❌ por poco | hueco 0,729; tras robo, invertida (va con retraso) |
| T2 | F2 clusters de situaciones de juego (E077) | 26 oct | ❌ | hueco 0,67; tras robo 0,68 |
| T2 | F3 firmas del robo y la contrapresión (E058) | 27 oct | ✅ puerta; ✅ **E058b pasa** (0,7410, 9 de 10) → ⏳ combinación con U6 (E075b) | **AUC tras robo 0,902** (centroide hacia delante) |
| T2 | F4 pressing intensity (E072) | 26 oct | ❌ | tras robo 0,695 |
| T2 | F5 territorio e inclinación del campo (E059) | 26 oct | ❌ | hueco 0,664 |
| T2 | F6 estado del juego sin balón, Lang 2022 (E071) | 27 oct | ✅ por poco; ❌ E071b no pasa (0,7271, +0,002) | hueco 0,879 frente a 0,867 |
| T2 | familia «control» (E051) | 26 oct | ❌ | AUC tras un robo 0,744 < 0,80; se reevalúa junto al poseedor |
| T3 | U1 tipo de plano del broadcast → D (E055-E057) | 27 oct | ✅ puerta; ❌ E057 no pasa (0,7311, 6 de 10) | AUC 0,90 en la ventana pública; 0,91 en los huecos; el 60 % de los D cortos va con plano abierto |
| T3 | U2 repeticiones del broadcast → D | 28-30 oct | pendiente | — |
| T3 | U3 toques desde el balón del vídeo sin filtrar (E092) | 27 oct | ❌ | lift 1,38 < 1,5 (ATD: 2,24); el balón del vídeo es demasiado ruidoso |
| T3 | U4 balón fuera o en un punto de reanudación (E055) | 26 oct | ❌ | AUC 0,51-0,56 |
| T3 | U5 jugadores + equipo por color (portero, árbitro) (E074) | 27 oct | ✅ como pieza de U6 | equipo por color 0,926; 20,4 % árbitros o porteros |
| T3 | U6 quién conduce (E074 puerta, E075 banco) | 27 oct | ✅ puerta; ✅ **E075 pasa** (0,7692, 10 de 10); ✅ **estrés E075s pasa** (0,7605, 10 de 10); ❌ guardarraíl E075c (D cortos, L1) → ⏳ combinación con F3 (E075b) | acierto implícito 0,90 (dudosos 0,95, huecos 0,87); cobertura 0,40-0,57 |
| T3 | U7 pose (saque de banda, portero con balón, barrera) (E091) | 27-28 oct (GPU) | ⏳ pasada de pose en marcha (YOLO-pose, mosaicos ×2, ~6 h) | prueba: 20 personas por frame, 92 % con muñecas |
| T3 | U8 árbitro y asistentes | 28-30 oct | pendiente | — |
| T3 | U9 energía de movimiento (flujo óptico) | 28-30 oct | pendiente | — |
| T3 | U10 Ball Action Spotting (timebox de 1 día) | 4-7 nov | pendiente | — |
| T4 | A1 mención → equipo (E055) | 26 oct | ❌ | seguro 0,73; dudoso 0,50-0,52; hueco 0,58-0,60 |
| T4 | A2 palabras de evento (E055) | 26 oct | ❌ | 0,50-0,53 |
| T4 | A3 sonido del golpeo (E076) | 26 oct | ❌ | AUC 0,518 en los toques |
| T4 | A4 ritmo de habla / «replay» (E055) | 26 oct | ❌ | 0,51-0,55 |
| T4 | A5 grada: ataque de quién | 28-30 oct | pendiente | — |
| T4 | A6 marcadores de tiempo (E055) | 26 oct | ❌ | 0,53 |
| T4 | silbato y grada como parón (E029, E045) | — | ❌ | sin señal |
| T5 | M5a poseedor tipo PathCRF (E078, E078b) | 26 oct | ❌ solo, peor que LightGBM; combinado con E049, 0,7236 frente a 0,7254 (2 de 10) | equipo en hueco 0,728 frente a 0,842 |
| T5 | M5b Ball Radar en los huecos | 26 oct | ❌ | equipo en hueco 0,728 < 0,85 |
| T5 | M5c MS-TCN (E079, E079b) | 27 oct | ❌ combinado con E049, 0,7207 frente a 0,7254 (4 de 10) | acierto parecido al LightGBM |
| T5 | M5d CatBoost/XGBoost, multitarea | 4-7 nov | pendiente | — |
| T5 | M5e bagging | 7-9 nov | pendiente | — |
| T6 | combinación, ablación, artefacto, congelación | 8-14 nov | pendiente | — |
| equipo | email Q2/Q3/Q6 (Q6 desbloquea el audio y el broadcast en la entrega) | ya | **Manel** | — |

## De dónde partimos y cuánto falta
- **Banco v3**: 0,7206 con la decodificación de la entrega. **Reto estimado**: ~0,62-0,68. **Objetivo**: ≥ 0,75 en
  el reto, unos 0,79 en el banco si la distancia entre los dos fuera la de siempre.
- **El margen medido** (oráculos, que no se suman):

  | qué | techo en el banco |
  |---|---|
  | ver los robos que no vemos | +0,146 (con balón visible +0,097, en hueco +0,054) |
  | no inventar cambios | +0,090 |
  | huecos perfectos | +0,098 |
  | decodificación | +0,001 |
  | fronteras de lo que ya vemos | +0,015 |

- **Lo que el banco no ve**: en el reto el modelo mete unos seis veces más D cortos que en el banco (E041), por un
  defecto del simulador, el balón lento (E043). **Inyectados en el banco a la tasa del reto, bajan la nota de 0,721
  a 0,633 (−0,088)**; a la mitad de la tasa, −0,047 (E040).
- **Conclusión**: no queda nada que ganar afinando lo que tenemos. Hacen falta **(1) arreglar el simulador para
  que el reto se parezca al banco y (2) una señal nueva para los robos cortos**.

## Mejoras, en orden

### M1. Banco v3.1: balón lento realista en el simulador (+ guardarraíl de forma en el reto)
- **Ataca**:
  - los D cortos del reto (0,68 por minuto frente a 0,09);
  - los robos que se convierten en «parón»;
  - los cambios directos que faltan en los huecos.
- **Evidencia**:
  - E042: la atribución señala `ball_speed` y `touch_rank`;
  - E043: tramos de balón lento ≥ 1 s por minuto de juego vivo: limpio 1,08, degradado 0,01, reto 0,55;
  - el error correlado del balón (`ball_corr_noise` 0,8 m, τ 75) es un proceso de Ornstein-Uhlenbeck sin suavizar
    que mete unos 4 m/s de velocidad aparente.
- **Coste estimado**: inyectado en el banco, el defecto cuesta **−0,088** a la tasa del reto y −0,047 a la mitad
  (E040). En el reto, quitarlo debería devolver **del orden de +0,05 a +0,09**.
- **Pasos**:
  1. `io/degrade._ball_signal`: **suavizar el error correlado** (filtrar el ruido de Ornstein-Uhlenbeck o usar un
     proceso de 2.º orden), de modo que conserve la deriva lenta (las patadas aplanadas de E013) pero no meta
     velocidad aparente.
     - Revisar también `_reveal_gaps`: interpola los parones en línea recta, la sospecha de E060.
     - Recalibrar con los objetivos de E013 (residuo 0,04 m; dv p50/p90/p99 1,8/5,6/11,8; 0,04 toques/s) **más dos
       nuevos**: % de frames lentos en juego vivo y tramos lentos ≥ 1 s por minuto, cerca de los del limpio
       (8,8 % y 1,08) y nunca por debajo de los del reto medidos sobre el vivo predicho (5,6 % y 0,55; el valor
       real es mayor, porque parte de los D cortos es vivo).
  2. Revisar `recv05_*`/`recv10_*`: toman z +4 a +13 con el balón lento (E042).
  3. Recalcular las features del banco (la caché se invalida sola), `tune_l3` con el protocolo y nueva referencia
     del banco v3.1. ADR-0017 («banco v3.1»: cambia el simulador e invalida las comparaciones con v3).
  4. **Guardarraíl en el reto (ADR-0018, pre-registrado antes de ver el resultado)**: con el artefacto nuevo,
     `real_vs_bench` debe dar:
     - tramos D < 5 s ≤ 0,25 por minuto (hoy 0,68);
     - tramos D 0,8-1,3 por minuto (hoy 1,78);
     - cambios directos ≥ 1,7 por minuto (hoy 1,40).

     Solo es una comprobación de paso o fallo con agregados de nuestra predicción, sin GT. No se usa para afinar
     parámetros.
  5. Artefacto nuevo, `make_submission` dos veces con el mismo sha y humo en la ventana pública.
- **Riesgo**:
  - el banco puede no subir (no tenía el defecto), y está bien: se acepta si no baja más de 0,005 y pasa el
    guardarraíl;
  - con el balón lento de vuelta, el modelo puede perder parte de su detector de parones; lo vigila el % de D.
- **Coste**: 1 día de código y calibración, unas 3 h de caché y evaluación, y el artefacto. **26-28 oct.**

### M2. Señal de control del balón para los robos cortos con el balón visible
- **Ataca**:
  - los 480 cambios no vistos con balón visible (+0,097);
  - los 467 sobrantes (+0,090);
  - las posesiones de 0 pases (las encontramos el 52 %) y las de menos de 3 s (43 %).
- **Evidencia**:
  - CHALLENGE.md define la posesión como **control**;
  - PathCRF y Ball Radar modelan **al jugador poseedor** antes que al equipo;
  - en el banco, «el jugador más cercano es el poseedor» solo acierta el 0,715 (0,779 con el balón limpio): la
    cercanía de un instante no es control;
  - tras un robo, el nivel de forma engaña y solo informa el cambio (E046).
- **M2a, la versión barata (1 día)**: features de **control por equipo** con el balón visible:
  - tiempo seguido con el balón a < 1,5 m de un jugador del equipo, y tiempo acumulado en los últimos 1, 2 y 3 s;
  - número de jugadores distintos del equipo que lo han «tenido» en 3 s (aproximación a los pases);
  - presión del rival en ese momento;
  - el cambio entre antes y después, como en las familias.

  **Puerta** (como E046): AUC en los 0-3 s tras un robo ≥ 0,80 (hoy, jugador más cercano 0,71 y modelo 0,86) y que
  la señal no tenga desajuste entre reto y banco (`feature_shift` < 0,25). Si pasa, evaluación completa.
- **M2b, la versión cara (4-5 días, solo si M2a se queda corta)**: modelo de **poseedor individual** con CRF
  enmascarado, tipo PathCRF. Usa el código público y lo adapta a 46 slots sin identidad, a D y al banco
  degradado.

  **Puerta** en 3 pliegues: cambios H↔A con frontera a ≤ 2 s ≥ 70 % (hoy 63 %) y sobrantes ≤ 350 (hoy 467),
  antes de la evaluación completa. Entra como fuente de probabilidades en la capa 2, como el modelo profundo de
  E034.
- **Riesgo**: el balón filtrado del ATD (sin toques) limita cualquier señal de control. Por eso M1 va antes:
  devuelve el balón lento, que es justo el balón controlado.

### M3. Más partidos de entrenamiento
- **Ataca**: todo, porque el modelo sigue limitado por datos (E010). Se pasó de 3 a 10 partidos con ganancia.
- **Fuentes**:
  - **SkillCorner open data**: 10 partidos, licencia MIT, poseedor por frame (`possession.group`), 10 fps. Hay que
    subirlo a 25 fps y deducir el D de sus fases de juego o eventos;
  - **PFF FC Mundial 2022**: 64 partidos, gratis con formulario; licencia y campos por confirmar.
- **Pasos**:
  1. confirmar licencias y las bases (Q3);
  2. cargador con kloppy y GT;
  3. degradación con los mismos objetivos;
  4. **curva de aprendizaje antes de invertir**: entrenar con 5 y con 9 partidos del banco v3.1. Si de 5 a 9 sube
     menos de 0,01, más datos no compensan.
- **Coste**: 2-3 días. **Responsable propuesto**: persona B (descarga y cargador) con el agente.

### M4. Ensamblado de semillas y calibración
- Barato. Ganancia esperada +0,003-0,01. **7-9 nov**, sobre el modelo que quede.

### M5. Balón inferido por los jugadores (Ball Radar) en los huecos
- **Ataca**: los huecos (techo +0,098; la mitad +0,047) y los 233 cambios no vistos en hueco.
- **Puerta**: en los huecos simulados del banco, el equipo del más cercano al balón inferido acierta ≥ 85 % en
  juego vivo (hoy el modelo acierta el juego vivo en hueco un 67-69 %).
- **Coste**: 2-3 días. Solo si M1 y M2 dejan tiempo.

### M6. YOLO como evidencia blanda en los huecos (revisión tras M1)
- E044: el equipo más cercano al balón del vídeo coincide en el **92,4 %** cuando el modelo está seguro, igual que
  el balón del ATD.
- Hoy no tiene recorrido: el 91 % de los huecos sale D y solo hay 3,9 min de juego vivo predicho en hueco.
- Si M1 devuelve juego vivo a los huecos, se mide otra vez el recorrido: es una feature `video_team` con la
  confianza, no el relleno de ADR-0016.

## Cerrado (con número)
| línea | motivo |
|---|---|
| Audio: silbato y grada (E029, E045) | Sin señal ni con PANNs (AUC 0,51-0,54) |
| Afinar el Viterbi, HSMM o duración mínima | +0,001 / 0 (E040) |
| Ajuste fino de fronteras (tipo ASRF) | techo +0,015 a ≤ 1 s (E040) |
| Más compacidad o formaciones (plantillas con y sin balón, SoccerCPD) | ya explotado: el modelo 0,965 frente a 0,87 de la mejor señal de forma; SoccerCPD trabaja a escala de minutos (E046) |
| Relleno de balón con YOLO (ADR-0016) | rechazado (E061-E069) |
| Regla «quitar D cortos» como parche | en el banco no ayuda (0 a −0,008, E040) y elegirla en el reto sería ajustar a ciegas. Se ataca la causa (M1) |

## Calendario
| fechas | agente (Manel) | persona B | persona C |
|---|---|---|---|
| 26-28 oct | **M1** simulador + evaluación + guardarraíl + artefacto | revisar la PR `analisis-fallos-v3`; empezar la descarga de SkillCorner y PFF | **reproducibilidad cruzada** del artefacto vigente (urgente) |
| 29 oct-1 nov | **M2a** features de control + puerta + evaluación | cargador de M3 y GT | documento técnico (borrador sobre E070) |
| 2-6 nov | M2b (si hace falta) o M3 (curva de aprendizaje y entrenamiento) | M3 | documento técnico |
| 7-9 nov | M4 semillas; M5/M6 si hay tiempo | — | — |
| 10 nov | **inscripción de los 3 confirmada** | | |
| 10-14 nov | artefacto final, dos veces el mismo sha, clon limpio en otra máquina | revisión final | documento técnico → PDF |

## Cuánto podemos esperar (orientativo)
| mejora | banco | reto |
|---|---|---|
| M1 | ~0 (el banco no tenía el defecto) | **+0,05 a +0,09** (coste inyectado del defecto, E040) |
| M2 | +0,02 a +0,05 si la señal pasa la puerta | parecido |
| M3 | +0,01 a +0,03 | parecido |
| M4 | +0,003 a +0,01 | parecido |

**Lectura honesta**: llegar a 0,75 en el reto pide que **M1 recupere lo que cuestan los D cortos y que M2 funcione**.
Con solo M1, el reto debería acercarse al banco (~0,70-0,72). Sin M1, nada de lo que mejore el banco se nota del
todo en el reto: el defecto se come casi 0,09.
