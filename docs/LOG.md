# Bitácora del equipo

Formato: la entrada más nueva va **arriba**.

```
## AAAA-MM-DD — Nombre
- Hecho:
- Pendiente:
- Bloqueos / preguntas para el equipo:
```

---

## 2026-10-22 — Manel (con agente)
- Hecho:
  - **Explicación completa del proyecto desde cero** → `docs/01_contexto_completo.md` (reto, datos, validación, modelo, resultados, fallos, futuro). Es el punto de entrada para compañeros y agentes nuevos.
  - Mañana: E017 (robustez con aleatorización), E018 (**apilado aceptado, 0,542**), E019 (la aleatorización no pasa el protocolo sobre el apilado; apilado desplazado 0,558). **ADR-0011**: modelo final = L3 apilado. `train_l3_final --stack` y `make_submission` ya soportan las dos capas (commit `1ab9dc2`).
  - **Análisis a fondo de fallos (E020)** → `docs/investigacion/02_analisis_fallos.md`, script `pipeline/failure_analysis.py`.
  - **Hallazgo**: la GT del banco contaba cada robo de 1 s, y el reto solo cuenta la posesión **controlada** (CHALLENGE.md). Con «un robo cuenta si dura ≥ 1,5 s» el banco reproduce la GT pública (13,7 s y 17,5 % < 3 s, frente a 14,2 s y 17,6 %), y el mismo modelo pasa de 0,542 a **0,628**. **ADR-0012** (propuesta), `external.controlled_possession`, `tune_l3 --controlled` y `--from-oof`.
  - Fallos que quedan: no vemos la mitad de los robos (79 % de las pérdidas), huecos largos (acierto 0,78), duelos, aéreos y despejes, infrasegmentación.
  - Idea del audio del broadcast (silbato y comentarios) → **ADR-0013** (propuesta, bloqueada hasta saber si hay audio) y preguntas **Q6/Q7** en `docs/00_reto.md`.
  - `docs/03_plan_ganar.md`: nueva sección «Estado y prioridades» con la tabla de tareas por persona.
- Pendiente:
  - E021: apilado reentrenado con la GT controlada (lanzado) y robustez con T = 1 y 2 s. Después, aceptar o rechazar ADR-0012.
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
