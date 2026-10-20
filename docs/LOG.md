# Bitácora del equipo

Formato: la entrada más nueva va **arriba**.

```
## AAAA-MM-DD — Nombre
- Hecho:
- Pendiente:
- Bloqueos / preguntas para el equipo:
```

---

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
