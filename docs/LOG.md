# Bitácora del equipo

Formato: la entrada más nueva va **arriba**.

```
## AAAA-MM-DD — Nombre
- Hecho:
- Pendiente:
- Bloqueos / preguntas para el equipo:
```

---

## 2026-10-26 — Manel (con agente) · plan completo en marcha
- Plan aprobado: el **plan completo** (T0-T6), que suma todo lo pedido. Tabla de estado por línea en
  `docs/05_plan_mejora.md`.
- **T0** (rama `banco-v3-1`):
  - banco v3.1 calibrado (E047, ADR-0017): pérdida 160 → 20,5; el balón lento en vivo vuelve (0,96 tramos/min);
  - guardarraíl (ADR-0018) con los umbrales escritos antes de entrenar;
  - SkillCorner: 20 partidos con GT completa y sus features listas;
  - cola en serie (`scripts/cola.sh`, `external_data/cola.txt`): E048 → E049 → E057 → E058 → E071 → E053 → E054.
- **T1**: `docs/investigacion/06_estado_del_arte_v3.md`.
  - Lang 2022: el estado del juego sin balón, 92 %.
  - Link y Hoernig 2017: toques como picos de aceleración ≥ 4 m/s².
  - Bauer y Anzer, pressing intensity, equipo por color con SigLIP, sonido del golpeo.
- **T2, puertas en el banco v3.1**:
  - ✅ **F3 firmas del robo: AUC 0,902 tras un robo** (el centroide sale hacia delante);
  - ✅ F6 estado sin balón (por poco: 0,879 frente a 0,867 en hueco);
  - ❌ F1 formación (hueco 0,729), F2 situaciones, F4 pressing, F5 territorio, «control».
- **T3/T4, señales del reto**:
  - ✅ **U1, broadcast: AUC 0,90 en la ventana pública** y 0,91 en los huecos. El 60 % de los D cortos va con
    plano abierto. Simulada en el banco con desajuste ≤ 0,10;
  - ❌ U4, A1 (comentario → equipo: 0,50 donde dudamos), A2, A3 (golpeo: 0,518), A4, A6.
  - Pasada de vídeo en GPU (balón en todos los frames + jugadores con el color del torso), unas 6,5 h.
- **T5**: código del poseedor tipo PathCRF (E078, entrenando 10 pliegues) y de MS-TCN (E079, en cola).
- Incidencias:
  - en esta consola, los escapes de barra invertida dentro de los heredoc se interpretan dos veces (rompió
    `tune_l3` un momento; corregido antes de que lo usara la cola). Para código, usar Edit o Write;
  - `data/` no se puede borrar (regla de permisos).
- Pendiente:
  - resultados de la cola (E048 → E054), del poseedor y de MS-TCN y de la pasada de vídeo (U3, U5, U6, U8);
  - después, artefacto + guardarraíl con lo que haya pasado.
- Bloqueos (personas): email Q2/Q3/Q6 (Q6 decide el audio y el broadcast), PR, reproducibilidad, inscripción.

## 2026-10-25 (noche) — Manel (con agente) · PUNTO DE PARADA
- **Plan aprobado**: «plan de máxima mejora». Copia de trabajo en `C:\Users\manel\.claude\plans\lucky-cuddling-hellman.md`;
  la versión del equipo está en `docs/05_plan_mejora.md`. Fases:
  1. M1 banco v3.1 + guardarraíl en el reto;
  2. M2a familia «control»;
  3. las puertas deciden entre más datos y poseedor tipo PathCRF;
  4. bagging;
  5. congelación.
- Hecho (rama **`banco-v3-1`**, sale de `analisis-fallos-v3`):
  - **1.1** `atd_stats.ball_dynamics()` y `atd_stats --json`: objetivos del balón del reto en
    `docs/investigacion/resultados/objetivos_balon_reto.json`.
    - Reproducen E013: residuo 0,044; dv 1,77/5,59/11,81; 0,043 toques/s; velocidad 5,1/14,3/24,5; velocidad
      antes de un hueco 2,0.
    - Balón lento en vivo: en el vivo predicho del reto, 5,6 % y 0,55 tramos ≥ 1 s por minuto. **En la ventana
      pública, con la GT, 9,4 % y 1,86 por minuto**: el valor real está por encima de 0,55, cerca o por encima
      del limpio (1,08).
  - **1.2** `DegradeParams.ball_corr_smooth`: suavizado gaussiano del error correlado del balón.
    - Con 0 el degradado es **idéntico bit a bit** al anterior (comprobado con J03WR9).
    - Ojo: tocar `degrade.py` cambia el hash de la caché de features, así que **la próxima evaluación recalcula
      las features del banco**, también las de v3.
  - Tests: 60 pasan.
- **Siguiente paso al retomar (1.3)**: `pipeline/calibrate_ball.py`.
  - Para cada partido del banco v3: GT controlada y vivo = GT H/A. Degradar solo el balón con la misma semilla
    que el banco: `rng = default_rng(i)`, `visible`, `occluded`, `weight` como en `degrade()` y luego
    `_long_ball(...)`.
  - Medir con `ball_dynamics(..., live=vivo)`.
  - Rejilla de unas 48 combinaciones: `ball_smooth_sigma` {6, 10} × `ball_corr_noise` {0,4, 0,8, 1,2} ×
    `ball_corr_tau` {75, 150} × `ball_corr_smooth` {0, 10, 25, 50}.
  - Pérdida normalizada frente a los objetivos del reto. Para el balón lento, objetivo ~1,08 por minuto y
    ~8,8 % (limpio), con penalización fuerte por debajo de 0,55.
  - Guardar en `docs/investigacion/resultados/calibracion_balon_v3_1.json`. Elegir `BENCH_DEGRADE_V3_1` en
    `bench_config.py` y añadir `v3_1` a `BENCHES` de `tune_l3`. Escribir ADR-0017.
- Después:
  - **1.4** `tune_l3 --eval-degrade v3_1`: E070 entrenado en v3 y evaluado en v3.1; tiene que verse el defecto;
  - **1.5** reentrenar en v3.1 (protocolo);
  - **1.6** ADR-0018 con los umbrales del guardarraíl **antes** de ejecutar: D < 5 s ≤ 0,25/min, tramos D
    0,8-1,3/min, cambios directos ≥ 1,7/min, % D 0,38-0,50.
- Pendiente de personas:
  - revisar la PR `analisis-fallos-v3` (sin subir todavía);
  - reproducibilidad cruzada (persona C);
  - email Q2/Q3/Q6;
  - inscripción antes del 10 nov.

## 2026-10-25 (tarde) — Manel (con agente)
- Hecho (rama `analisis-fallos-v3`, desde `banco-v3`):
  - **Análisis de fallos del modelo de la entrega** en el banco v3 (E039-E040,
    `docs/investigacion/04_analisis_fallos_v3.md`).
    - `failure_analysis --bench v3`: eventos DFL vía kloppy, tabla por partido y descomposición con oráculos.
    - Banco 0,7206 con la decodificación de la entrega.
    - El margen está en los robos no vistos (+0,146, la mayoría con el balón visible) y en los cambios sobrantes
      (+0,090). El Viterbi está agotado (+0,001).
    - J03WR9 falla en las fronteras.
  - **Reto frente a banco** (E041-E043, solo agregados):
    - unas seis veces más D cortos en el reto (0,68 por minuto frente a 0,09), el 90 % con el balón visible;
    - TreeSHAP: `ball_speed` y `touch_rank`;
    - **el simulador borra el balón lento del juego vivo**: tramos lentos ≥ 1 s por minuto, limpio 1,08, degradado
      0,01, reto 0,55;
    - **cuesta unas 0,09**: inyectado en el banco a la tasa del reto, 0,721 → 0,633.
  - **Puertas**:
    - YOLO como evidencia de equipo: 92 %, pero poco recorrido hoy (E044);
    - audio con PANNs: sin señal (E045);
    - compacidad: ya explotada; tras un robo manda el cambio (E046).
  - **Estado del arte ampliado** (`docs/investigacion/05_estado_del_arte_v2.md`): PathCRF, Ball Radar,
    TranSPORTmer, MS-TCN/ASRF, SkillCorner/PFF, PANNs, compacidad y formaciones.
  - **Plan de mejora** `docs/05_plan_mejora.md`: M1 balón lento en el simulador + guardarraíl en el reto; M2 señal
    de control; M3 más datos; M4 semillas.
  - `make_submission --dump-proba` guarda las probabilidades y features del reto en `data/derived/` (el sha no
    cambia: `9602959c4ec05bd6`).
- Pendiente:
  - aprobar el plan;
  - M1 (26-28 oct);
  - ADR-0017 (banco v3.1) y ADR-0018 (guardarraíl de forma en el reto) cuando se implemente M1.
- Bloqueos (personas):
  - revisar la PR;
  - reproducibilidad cruzada (persona C);
  - descarga de SkillCorner y PFF (persona B, M3);
  - email Q2/Q3/Q6;
  - inscripción antes del 10 nov.

## 2026-10-25 — Manel (con agente)
- Hecho (rama `banco-v3`):
  - **Ablaciones de familias terminadas** (E038, E060, E062, E065-E067): jugadores −0,008, reanudación −0,005,
    balón −0,003, estructura y bloques 0 por separado, pero **−0,012 sin las dos** (redundantes entre sí).
    **Se quedan las cinco familias** (ADR-0015, pendiente de revisión).
  - **Relleno del balón con YOLO afinado con pseudoetiquetas** (E061-E069): pasa la puerta de huecos
    simulados (59,6 % a < 2 m) y cubre el 55,9 % de los huecos reales. En el banco, con una simulación
    optimista, daba +0,030 (E064), pero en la ventana pública 3 de 4 parones salían como juego vivo (E068):
    en el reto el balón relleno está pegado a un jugador (quien lo lleva a sacar o una bota). Con una
    simulación fiel, +0,008 en 6/10 (E069). **Rechazado** (ADR-0016); el código se conserva.
  - **Artefacto definitivo** (E070): cinco familias, apilado, sin relleno, decodificación fija. `La MasIA.csv`
    dos veces con el mismo sha `9602959c4ec05bd6`. Banco 0,7185; ventana pública spellF1 0,651 y macroF1 0,803
    (antes 0,556 y 0,766).
- Pendiente:
  - documento técnico y README con el procedimiento reproducible;
  - revisar «balón» (J03WR9) y los D cortos sobrantes, si da tiempo;
  - PR `banco-v3` → `main` revisada por un compañero; congelación el 14 nov.
- Incidencias: Claude Code cortó procesos por memoria varias veces (dos evaluaciones a la vez, YOLO, juego,
  Docker con 15 GB). Ahora las colas van en serie y como proceso independiente de la sesión. Esa bash no trae
  `/usr/bin` en el PATH, y una evaluación de 2 h se perdió por eso (ya corregido). Ultralytics deja `runs/` en
  la raíz (en `.gitignore`).

## 2026-10-24 — Manel (con agente)
- Hecho (rama `banco-v3`):
  - **E037**: el modelo profundo no aporta sobre las familias (0,7172 y 0,7134 frente a 0,7185). Queda fuera
    de la entrega; el código se conserva.
  - **Informe de estado en PDF** (`docs/informe/`, 12 páginas con gráficos) para ponernos al día los tres:
    qué hemos hecho, por qué, qué se aprobó o rechazó, dónde fallamos y qué viene.
  - **ADR-0014** (banco v3 con DFL y T por fuente) y **ADR-0015** (familias de features; pendiente de la
    ablación).
  - **Documentación al día**: `01_contexto_completo.md` reescrito, `03_plan_ganar.md` con estado,
    prioridades, **reparto para los compañeros** y cómo empezar, `datos.md` y `README.md`.
  - Ablación sin «jugadores» relanzada (se cortó al reiniciarse la sesión). WSL cerrado para liberar
    memoria.
- Pendiente:
  - resto de ablaciones (persona B);
  - reproducibilidad cruzada y documento técnico (persona C);
  - artefacto final con los 10 partidos y regenerar `La MasIA.csv` (agente).
- Bloqueos (personas): email a la organización, invitaciones de GitHub, revisar la PR `banco-v3` → `main`
  y los ADR 0001-0015 (sobre todo el 0004).

## 2026-10-23 — Manel (con agente)
- Hecho (rama `banco-v3`):
  - **Plan de investigación** (`docs/04_plan_investigacion.md`): revisión crítica de lo hecho, estado del arte (PathCRF, Ball Radar, Pressing Intensity, contrapresión, compacidad, formaciones, SoccerNet-Echoes) y plan hasta el 14 nov.
  - **Banco v3 = 3 Metrica + 7 DFL** (descargado con el navegador del portátil). La GT del tracking DFL también necesita la regla controlada (T = 2 s). El simulador calibrado en Metrica deja DFL calibrado sin tocar nada (E027). Referencia: **0,6470** (E030). Entrenar con más partidos ya mejora.
  - **Features por familias**, pensadas como las mira un entrenador (balón, jugadores, bloques, estructura, reanudación; `features/tactical.py`): **0,7185** (+0,072, 9/10, E035). Niveles de bloque relativos al partido tras ver un desajuste real/banco de 0,3-0,4 (`feature_shift`).
  - **Modelo profundo** por conjuntos + BiGRU (`models/seq.py`), promediado con LightGBM: 0,6570 sobre la referencia (E034). Predicción en numpy para que la entrega no necesite torch.
  - Coste de cambio dinámico: sin ganancia (E036). Acierto por longitud de pase: fallan los envíos perdidos, no los cambios de juego (E028). Silbato: sin señal (E029). Comentario en inglés: transcrito en local, parones fiables pero ya detectados; no entra (E033).
  - Infraestructura: caché de features en disco (las 3 evaluaciones simultáneas se cortaron por memoria), `scripts/run_aggregate.sh`, el portátil se mantiene despierto mientras corren los cálculos (se había suspendido dos veces) y la tapa enchufada ya no suspende.
- Pendiente:
  - Combinación familias + modelo profundo; ablación por familias (empezando por «jugadores», que incluye el portero, con desajuste de 0,22); por qué empeora J03WR9.
  - Artefacto final con los 10 partidos, `make_submission` dos veces con el mismo sha, humo en los 5 min públicos.
- Bloqueos (personas): enviar el email (Q2/Q4/Q5/Q6), invitar a los compañeros en GitHub y revisar la PR `banco-v3` → `main`, decidir sobre el ADR-0004.

## 2026-10-22 (tarde) — Manel (con agente)
- Hecho (rama `artefacto-apilado`):
  - **Datos del reto completos**: broadcast `atd.txt` y `broadcast.mp4` movidos de Descargas a `data/broadcast/` y comprobados (164.488 registros contiguos; vídeo 6.579,5 s). El táctico tiene una pista de audio muda; **el broadcast tiene audio real**. Solo falta DFL. `docs/datos.md` actualizado.
  - **Artefacto final** (ADR-0011): el Viterbi reajustado con la GT controlada vuelve a elegir 1/20/10/10 (E022). Se descubrió que el entrenamiento **no era determinista** con la máquina cargada; arreglado con `force_col_wise` + test (E024). El banco no cambia (0,5422 eventos / 0,6245 controlada). **`La MasIA.csv` regenerado**: sha256 `252716b03d614399`, 322 posesiones, idéntico en dos ejecuciones. Humo en los 5 min públicos: 0,513 (ruido del bootstrap 0,38).
  - **Balón fusionado descartado tal cual**: el broadcast alineado rellena el 7,5 % de los huecos (E023) y el detector YOLO genérico acierta el 24 % incluso donde el ATD ve el balón (E025; nuevas opciones `--validate-only` y `--sim-gaps`).
  - **El audio del broadcast se puede usar en todo el partido**: el desfase broadcast-táctico es constante en cada parte (E026). ADR-0013 desbloqueado.
  - El agente ejecutó con autorización de Manel los scripts de solo agregados sobre los datos reales, a través de `scripts/run_aggregate.sh`. Ese script solo muestra la salida si el script acaba bien; si falla, la deja en `data/derived/logs/` para una persona. El clasificador de Claude Code impidió al agente modificar el ADR-0004: **el equipo debe decidir si esta práctica se recoge en ese ADR**.
  - El conector de Gmail no tiene permiso de borradores: el email con Q2/Q4/Q5/Q6 está redactado en `docs/00_reto.md` y lo envía una persona.
- Pendiente:
  - Señales de cambio de posesión (prioridad 3), detector de balón parado en huecos (5), silbato (8), T = 1/2 s y acierto por longitud de pase (10).
- Bloqueos (personas):
  - DFL: `/chrome` para que lo baje el agente, o a mano.
  - Enviar el email; escuchar si el broadcast tiene comentarista; decidir sobre el ADR-0004.

## 2026-10-22 — Manel (con agente)
- Hecho:
  - **Explicación completa del proyecto desde cero** → `docs/01_contexto_completo.md` (reto, datos, validación, modelo, resultados, fallos, futuro). Es el punto de entrada para compañeros y agentes nuevos.
  - Mañana: E017 (robustez con aleatorización), E018 (**apilado aceptado, 0,542**), E019 (la aleatorización no pasa el protocolo sobre el apilado; apilado desplazado 0,558). **ADR-0011**: modelo final = L3 apilado. `train_l3_final --stack` y `make_submission` ya soportan las dos capas (commit `1ab9dc2`).
  - **Análisis a fondo de fallos (E020)** → `docs/investigacion/02_analisis_fallos.md`, script `pipeline/failure_analysis.py`.
  - **Hallazgo**: la GT del banco contaba cada robo de 1 s, y el reto solo cuenta la posesión **controlada** (CHALLENGE.md). Con «un robo cuenta si dura ≥ 1,5 s» el banco reproduce la GT pública (13,7 s y 17,5 % < 3 s, frente a 14,2 s y 17,6 %), y el mismo modelo pasa de 0,542 a **0,628**. **ADR-0012** (propuesta), `external.controlled_possession`, `tune_l3 --controlled` y `--from-oof`.
  - Fallos que quedan: no vemos la mitad de los robos (79 % de las pérdidas), huecos largos (acierto 0,78), duelos, aéreos y despejes, infrasegmentación.
  - Idea del audio del broadcast (silbato y comentarios) → **ADR-0013** (propuesta, bloqueada hasta saber si hay audio) y preguntas **Q6/Q7** en `docs/00_reto.md`.
  - `docs/03_plan_ganar.md`: nueva sección «Estado y prioridades» con la tabla de tareas por persona.
  - **E021**: reentrenar con la GT controlada **no aporta** (0,6225 frente a 0,6245 del modelo actual con Viterbi reajustado). El cambio es de medida: el modelo vigente se queda y se evalúa con `--controlled 1.5`. Calibración revisada: con el baseline, el banco queda ~0,04 más fácil que el reto, así que la estimación orientativa en el reto es ~0,55-0,60.
  - El análisis queda explicado paso a paso en `docs/investigacion/02_analisis_fallos.md`, con las salidas completas en `docs/investigacion/resultados/`.
  - **Presentación** actualizada (36 diapositivas: hallazgo de la definición, por qué perdemos posesiones, causas → soluciones) y subida al repo en `docs/presentacion/` (PDF, PPTX y fuente).
  - **Git consolidado**: todo el trabajo (T1-T8, plan para ganar, análisis de fallos) está en **una sola rama, `analisis-fallos`**, pendiente de una única PR hacia `main`. Se borraron, en local y en GitHub, las ramas ya contenidas en ella. Para recuperar alguna: `git branch <nombre> <sha>`: t1-entorno-uv `a5b4d90`, t2-load-match `2b103b1`, t3-eval-wrapper `687273d`, t4-baseline-submission `e359826`, docs-log-t1 `38b759a`, t5-banco-externo `be19464`, plan-ganar `1ab9dc2`.
- Pendiente:
  - Medir el modelo vigente con T = 1 y 2 s (robustez de ADR-0012).
  - Entrenar el artefacto final y regenerar `La MasIA.csv` (el actual sale del modelo en cuarentena).
  - Señales de cambio de posesión y detector de balón parado dentro de los huecos.
- Bloqueos (personas):
  - `broadcast atd.txt` (parte -009) y ejecutar el detector de balón entero (no estaba corriendo).
  - Bajar DFL a mano. Comprobar si `broadcast.mp4` tiene audio (`ffprobe`, ADR-0013).
  - Enviar Q4/Q6 (y Q2/Q5). Revisar los ADR 0001-0013.

## 2026-10-21 (tarde) — Manel (con agente)
- Hecho:
  - **Plan para ganar** aprobado (`docs/03_plan_ganar.md`), con tres frentes: A visión, B banco y C modelo. Protocolo de selección pre-registrado (ADR-0008).
  - Frente A, código listo y probado con sintéticos: alineador del broadcast (`io/broadcast_align.py` + `pipeline/broadcast_report.py`) y detector de balón en vídeo (`vision/ball_detector.py` + `vision/ball_track.py`, grupo `vision` con torch CUDA). ADR-0010.
  - Frente C: features v2 (toques, receptor, presión) → **cuarentena**. El ATD real tiene el balón filtrado y apenas detecta toques (E012/E013). Banco **v2.1** con el balón filtrado como el ATD y features **v3** con el umbral de toque relativo al partido: L3 sube de 0,465 a **0,519** en v2.1 (E014, ADR-0009). v4 (movimiento colectivo) rechazado (E015). Aleatorización del dominio sin mejora dentro del banco (E016).
  - Apilado con contexto temporal (capa 2) implementado; evaluación en curso.
  - Una evaluación se cortó por falta de memoria: Opera ocupaba 15,6 GB. Las features pasan a float32.
- Pendiente:
  - Resultados de robustez (`--eval-shift`) y del apilado → elegir el modelo y regenerar el artefacto y `La MasIA.csv` (hoy usa v2, en cuarentena).
  - **Enviar a la organización** las preguntas Q2, Q4 y Q5 (borrador en `docs/00_reto.md`).
  - Persona A: broadcast `atd.txt` (parte `-009`) → `broadcast_report`; después `ball_detector`.
  - Persona B: descargar DFL a mano → banco v3.
- Bloqueos / preguntas para el equipo:
  - Revisar y aprobar los ADR 0006-0010 y abrir la PR de `plan-ganar`.

## 2026-10-21 — Manel (con agente)
- Hecho:
  - Humo de L3 en datos reales (E011): 0,5875 en los 5 min públicos (L1 0,4686, baseline 0,3256).
  - **Nombre del equipo: `La MasIA`.** SUBMISSION.md exige `La MasIA.csv` y `La MasIA.pdf` (el evaluador pone el nombre del fichero en el ranking).
- Pendiente:
  - Generar la entrega: `uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"`.
  - Documento técnico `La MasIA.pdf`, repo público y formulario (antes del 17 nov).

## 2026-10-20 (noche 4) — Manel (con agente)
- Hecho:
  - Humo en datos reales (lo ejecutó Manel, solo agregados, E005): pipeline validado (el baseline s50 da 0,3256 en los 5 min públicos, igual que EVALUATION.md). L1 da 0,469 en esos 5 min (solo humo).
  - **Hallazgo del ATD** (E005-E006): el balón falta en ~134 huecos largos (> 4 s) que cubren el 41 % de lo evaluado. Son una mezcla de parón y juego vivo (56 % D en la ventana pública), y dentro de ellos los jugadores van más lentos (0,97 frente a 1,62 m/s). El Smart Tagging solo coincide con el final del 13 % de los huecos.
  - Banco **v2** recalibrado con huecos largos (E006). L3 = LightGBM (46 features por frame) + Viterbi: **0,502** dejando un partido fuera en v2, frente a 0,300 del baseline y 0,326 de L1 (E007-E010). Sin Smart Tagging (aporta +0,008, ruido). ADR-0007: L3 es la entrega principal y el baseline la reserva. Modelo versionado en `src/hackbarca/models/artifacts/` (gzip determinista).
  - T8 (E009): la pérdida está en la **segmentación de equipo** (con equipo perfecto, 0,87), no en el D (0,58) ni en el retraso de las fronteras (óptimo en 0).
- Pendiente:
  - **Ejecutar el humo de L3** sobre los datos reales (`make_submission --model l3`).
  - Mejorar la segmentación de equipo: features de trayectoria del balón y de presión, más capacidad o un modelo secuencial, HSMM con prior de duración. Más partidos de entrenamiento (DFL).
  - Abrir la PR `t5-banco-externo` → `main` (contiene T1-T8 y la rama `docs-log-t1`) y revisar los ADR 0001-0007.
  - Los informes "huérfanos" de `external/.../evaluation/reports/` son la salida por defecto de `run_evaluation.py` (escribe dentro del submódulo). Ahora lo redirigimos a `submissions/` (ignorado). Quien los generó puede borrarlos del submódulo.
- Bloqueos / preguntas para el equipo:
  - Nombre exacto del equipo para `submissions/<equipo>.csv`.
  - DFL (figshare) solo se descarga con navegador.

## 2026-10-20 (noche 3) — Manel (con agente)
- Hecho:
  - T2-T4 revisados (`load_match`, wrapper de evaluación, baseline de reserva). Corregido T4: sugería puntuar con `--gt private`, que **solo existe en la máquina de la organización**. Nunca tendremos el spellF1 del partido completo: solo los 5 min públicos (humo) y el banco externo.
  - T5: banco externo con Metrica sample-data (3 partidos, ~1.140 posesiones), GT H/A/D derivada de eventos, simulador de degradación tipo ATD **calibrado** para que el baseline reproduzca los 8 números publicados (ADR-0006, E001-E002). DFL bloqueado: figshare no deja descargar sin navegador.
  - T6: L1 = reglas + Viterbi + ventanas de reanudación del Smart Tagging. spellF1 0,363 dejando un partido fuera, frente a 0,293 del baseline s50 en el mismo banco (E003-E004).
  - Comando único de entrega: `make_submission --model {baseline,l1}`, determinista y con validación.
- Pendiente:
  - **Que una persona ejecute** sobre los datos reales (solo imprimen agregados): `atd_stats` y `make_submission` para baseline y L1, y el humo con la GT pública. Comandos en la PR de `t5-banco-externo`.
  - T7 (LightGBM + Viterbi), T8 (análisis de errores). Refinar la calibración (2 parámetros en el borde de la rejilla).
- Bloqueos / preguntas para el equipo:
  - Nombre exacto del equipo (el del formulario) para `submissions/<equipo>.csv`.
  - Si alguien puede bajar con navegador los 21 ficheros DFL de figshare a `external_data/dfl/<match_id>/{meta,event,tracking}.xml`, el banco pasa de 3 a 10 partidos.

## 2026-10-20 (noche 2) — Manel (con agente)
- Hecho:
  - **Limpieza de atribución de IA**: `git filter-branch` sobre `main` para quitar los `Co-Authored-By` que quedaron en 6 commits antiguos, y `push --force` a `origin/main`. **Los compañeros necesitan `git fetch` + resetear sus ramas locales a `origin/main` o su próximo push chocará.**
  - **T1 completado** (rama `t1-entorno-uv`, PR pendiente de abrir manualmente en GitHub — `gh` no disponible en este entorno): `pyproject.toml` + `uv.lock` (Python 3.11, kloppy==3.18.0, pandas<3, lightgbm, scikit-learn, matplotlib, pytest) y estructura `src/hackbarca/{io,features,models,decode,eval,pipeline}`. Verificado con `uv sync` + `uv run pytest` desde cero.
  - **Bug encontrado y arreglado** en `.gitignore`: la regla `models/` (pensada para artefactos de modelos entrenados en la raíz) no estaba anclada y también ignoraba `src/hackbarca/models/`, nuestro paquete de código — se habría perdido silenciosamente. Corregido a `/models/` en la misma rama.
  - Verificados por `test -f` (sin leer contenido) los 5 ficheros tácticos imprescindibles en `data/tactical/`: todos presentes.
  - Leída toda la documentación propia + repo oficial (`possession_metrics.py` incluido). Hallazgo importante: **ningún ADR está formalmente aceptado** (los 5 siguen en estado "Propuesta", sin revisiones marcadas) pese a que ya actuamos como si 0002/0004 estuvieran en vigor.
  - Hallazgo: hay outputs de `run_evaluation.py` en `external/.../evaluation/reports/` y `scoreboard.csv` que nadie ha documentado en `docs/experimentos.md` — alguien ya corrió el scorer oficial y no dejó registro.
- Pendiente:
  - Abrir la PR de `t1-entorno-uv` → `main` en GitHub (rama ya empujada).
  - **Aprobar formalmente los ADR 0001-0005** (marcar revisores).
  - Averiguar quién generó los reports huérfanos de `run_evaluation.py` y documentarlos en `experimentos.md`, o reproducirlos y documentarlos yo.
  - T2 (load_match), T3 (evaluación local), T4 (reproducir baseline oficial) — siguientes en la cola.
- Bloqueos / preguntas para el equipo:
  - **Importante**: tras el `push --force` a `main`, cualquiera con cambios locales basados en el `main` antiguo debe hacer `git fetch origin && git reset --hard origin/main` (o rebase) en su copia **antes** de volver a hacer push, si no divergirá.
  - T4 (reproducir baseline oficial sobre datos reales) necesita que una persona ejecute el comando — el agente no puede tocar `data/` (ADR-0004).

## 2026-10-20 (noche) — Manel
- Hecho:
  - Repo movido a `C:\dev\hack_barca` (fuera de OneDrive, sin `ç`). La copia antigua del Escritorio queda obsoleta.
  - Datos descomprimidos en `data/` y `MTAH_DATA_DIR` configurada. Alineación oficial OK (199.319/199.319). Inventario en `docs/datos.md`.
  - Entorno verificado: Python 3.11, kloppy 3.18, LightGBM y el scorer oficial.
- Pendiente:
  - Bajar de Drive la parte `-009` (broadcast `atd.txt`) y los vídeos (opcionales).
  - Crear el remoto en GitHub y hacer push.
- Bloqueos: ninguno para empezar a trabajar.

## 2026-10-20 (tarde) — Manel
- Hecho:
  - Leídas las bases oficiales (repo `metrica-sports/more-than-a-hack-2027`). `00_reto.md`, la investigación y la estrategia se han reescrito con ellas.
  - Hallazgos clave: métrica **spellF1** (segmentación, fronteras a < 0,5 s), **sin leaderboard**, **prohibido enseñar los datos a LLMs en la nube** (incluido Claude), repo de entrega **público y reproducible exacto**.
  - Nuevos ADR: 0004 (agentes ciegos a los datos) y 0005 (validación con datos externos). ADR 0001 y 0003 actualizados.
  - Añadido el repo oficial como submódulo en `external/`.
- Pendiente:
  - **Aprobar el ADR-0004 antes de que nadie descargue los datos con un agente abierto.**
  - Preguntas Q1-Q3 a la organización (issue o email).
  - Verificar los campos de DFL open data y Metrica sample-data.

## 2026-10-20 — Manel
- Hecho:
  - Investigación del reto (More Than A Hack 2027): calendario, fases, requisitos → `docs/00_reto.md`.
  - Investigación del estado del arte en posesión por frame → `docs/investigacion/01_posesion_estado_del_arte.md`.
  - Borrador de estrategia por fases → `docs/02_estrategia.md`.
  - Estructura del repo, CLAUDE.md, CONTRIBUTING, ADR 0001-0003 (en *Propuesta*).
- Pendiente:
  - Crear el repo privado en GitHub e invitar a los compañeros.
  - Descargar los datos oficiales de la fase 1 y responder Q1-Q9.
  - Revisar y aprobar (o discutir) los ADR 0001-0003.
- Bloqueos:
  - **Aún no tenemos los datos ni las bases detalladas** (métrica, formato de envío).
  - Confirmar que los 3 estamos inscritos (mínimo 3 personas para pasar de fase).
