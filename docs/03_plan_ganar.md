# Plan para ganar la fase 1 (La MasIA)

> Aprobado el 2026-10-21. La sección «Estado y prioridades» manda; lo que hay desde «Contexto» hacia abajo
> es el plan original del 21 oct y se conserva como histórico. Plan de investigación vigente:
> `docs/04_plan_investigacion.md`. Explicación completa: `docs/informe/La_MasIA_informe_estado_2026-10-24.pdf`.


## Estado y prioridades (actualizado 2026-10-24)

**Dónde estamos**
- **Banco v3** = 3 Metrica + 7 DFL (ADR-0014). Referencia (apilado v3): **0,6470** (E030).
- **Candidato**: apilado + **familias de features** (balón, jugadores, bloques, estructura, reanudación):
  **0,7185** (+0,072, 9/10, E035, ADR-0015). Antes de entregarlo falta la ablación por familias.
- **Rechazados esta semana, con número**: coste de cambio dinámico (E036), modelo profundo sobre las
  familias (E037), comentario dentro del modelo (E033), silbato (E029), broadcast como balón (E023) y YOLO
  genérico (E025).
- **`La MasIA.csv` actual**: válido y reproducible (sha `252716b03d614399`), pero sale del modelo v3
  (0,6245 en el banco v2). Hay que regenerarlo con el candidato.
- Estimación orientativa en el reto con el candidato: ~0,62-0,68 (incierta).

**Prioridades, en orden**
| # | tarea | quién | estado |
|---|---|---|---|
| 1 | Ablación sin «jugadores» (incluye el portero, con desajuste real/banco de 0,22) | agente (Manel) | **en marcha** |
| 2 | Resto de ablaciones: sin balón, sin bloques, sin estructura, sin reanudación | **persona B** (en su máquina, en paralelo) | por hacer |
| 3 | Análisis de fallos con el modelo nuevo en el banco v3 (hoy `failure_analysis` solo cubre Metrica) y por qué empeora J03WR9 | persona B + su agente | por hacer |
| 4 | Decidir la composición final (ADR-0015 → *Aceptada*); artefacto con los 10 partidos (`train_l3_final --bench v3`) | agente (Manel) | tras 1-3 |
| 5 | Regenerar `La MasIA.csv` (dos veces, mismo sha) + humo con los 5 min públicos | agente (Manel) | tras 4 |
| 6 | **Reproducibilidad cruzada**: en otra máquina, reentrenar el artefacto y regenerar el CSV, y comparar los sha | **persona C** | por hacer (urgente antes del 14) |
| 7 | Documento técnico: revisar y completar `docs/entrega/La MasIA - technical document.md` → PDF | **persona C** | por hacer |
| 8 | Mejoras baratas: ensamblado de semillas, calibración, reelegir el Viterbi en v3 | agente | 7-9 nov |
| 9 | Email a la organización (Q2/Q4/Q5/Q6), invitar a los compañeros en GitHub, revisar los ADR 0001-0015 | **Manel** | pendiente |
| 10 | PR `banco-v3` → `main` revisada y aprobada por otra persona (ADR-0001) | **persona B o C** | pendiente |
| 11 | Congelar el 14 nov; repo público; subir CSV y PDF al formulario el 15-16 nov | los tres | 14-16 nov |

## Cómo empezar (compañeros)
1. **Leer**: este apartado, `docs/01_contexto_completo.md` y el PDF de `docs/informe/` (30-40 min).
2. **Instalar** (abrid Claude Code **desde la carpeta del repo**, para que se aplique el bloqueo de `data/`):
   ```
   git clone --recursive https://github.com/ManelCM/FCBCN_HACK_2027.git C:/dev/hack_barca
   cd C:/dev/hack_barca && git checkout banco-v3 && uv sync && uv run pytest
   ```
3. **Datos**:
   - **del reto**: del Drive de la organización a `data/`, con `MTAH_DATA_DIR` apuntando ahí. Ningún agente
     los lee (ADR-0004).
   - **públicos**: Metrica se baja solo. DFL: abrid en el navegador las 21 URLs
     `https://springernature.figshare.com/ndownloader/files/<id>` (los id están en
     `kloppy/_providers/sportec.py`, `get_IDSSE_url`) y luego
     `uv run python -m hackbarca.io.dfl --from-downloads <carpeta de descargas>`.
4. **Reglas**:
   - nada de `data/` en git;
   - todo experimento en `docs/experimentos.md`, también los que fallan;
   - cada sesión, una entrada en `docs/LOG.md`;
   - cada decisión no trivial, un ADR;
   - rama propia y PR.
   - **Numeración de experimentos para no pisarnos**: Manel E038-E059, persona B E060-E079, persona C
     E080-E099.

### Persona B: ablaciones (prioridad 2)
- Cada evaluación tarda 1-2 h y pide unos 8 GB de RAM libres (cerrad WSL/Docker). No lancéis más de una a la
  vez. Se compara contra 0,7185 por partido y en el agregado.
  ```
  uv run python -m hackbarca.pipeline.tune_l3 --bench v3 --no-tags --stack --controlled 1.5 --controlled-dfl 2 --train-gt raw --jobs 2 --features <FAMILIAS>
  ```
- `<FAMILIAS>`, una por evaluación:
  - `touch+players+blocks+structure+restart` (sin balón);
  - `touch+ball+players+structure+restart` (sin bloques);
  - `touch+ball+players+blocks+restart` (sin estructura);
  - `touch+ball+players+blocks+structure` (sin reanudación).
- El entrenamiento es determinista: si en vuestra máquina repetís la del candidato
  (`touch+ball+players+blocks+structure+restart`), debería salir exactamente 0,7185. Es una buena prueba de
  que el entorno está bien.

### Persona C: reproducibilidad y documento técnico (prioridades 6-7)
- **Prueba cruzada**:
  1. `uv run python -m hackbarca.pipeline.train_l3_final --features 3 --no-tags --stack --temperature 1 --switch 20 --to-dead 10 --from-dead 10`
     → los sha256 de `src/hackbarca/models/artifacts/` tienen que coincidir con los del repo
     (capa 1 `8e44cc9d…`, capa 2 `80137a05…`).
  2. `scripts/run_aggregate.sh sub uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"`
     → sha `252716b03d614399`. Si no coincide, es lo más urgente del proyecto.
- **Documento técnico**: el borrador está en `docs/entrega/`; el informe de `docs/informe/` tiene todo el
  material.

## Contexto (plan original del 21 oct, histórico)
- **Estado (21 oct):** L3 (LightGBM + Viterbi) saca 0,502 de spellF1 en el banco externo v2, frente a 0,300 del baseline, y 0,588 en los 5 min públicos (comprobación). El baseline oficial da 0,243 en el partido completo real. El banco es ~0,05 más fácil que el reto, así que estimamos ~0,40-0,45 en el reto. Seguramente basta para el top 20; para ganar, no.
- **Dónde se pierde** (E009): con el equipo perfecto pasaríamos a 0,87; con el D perfecto, solo a 0,58. En el 41 % del partido falta el balón (134 huecos largos), y ahí el acierto cae de 0,89 a 0,74. Además estamos limitados por datos: solo 3 partidos de entrenamiento y la calidad aún sube con más capacidad (E010).
- **Restricciones:** los agentes no ven los datos del reto (ADR-0004); los modelos que procesen vídeo o tracking corren en local en vuestra máquina. Reproducción exacta con un solo comando. Prohibido etiquetar a mano. Sin leaderboard, así que el banco es nuestro único juez.
- **Equipo:** 3 personas a dedicación media (~8-12 h/semana cada una) más el agente, que escribe casi todo el código. Recursos: RTX 3060 de 6 GB y el vídeo táctico 1080p con homografía válida en el 100 % de los frames.

## Qué nos diferencia de los demás
1. **Rellenar los huecos del balón** con la segunda cámara (tracking broadcast) y con un detector local sobre el vídeo táctico. Casi nadie lo hará y ataca directamente la mayor fuente de error.
2. **Un banco de validación fiable** (10 partidos, la mayoría con GT de posesión sacada del tracking, como la del reto) para elegir con datos y no a ciegas.
3. **Modelar como mide spellF1**: secuencias y segmentos, con prior de duración y fronteras ajustadas a los toques, en lugar de clasificar frame a frame.

## Frentes de trabajo (en paralelo)

### Frente A: visión y fusión del balón (responsable: persona A, ejecuta en local)
1. **Descargar el `atd.txt` del broadcast** (la parte `-009` de Drive) y terminar de bajar el vídeo broadcast.
2. **Alinear el broadcast con el táctico** sin vídeo: ambos trackings están en coordenadas de campo. Por tramos entre cortes, buscar el desfase temporal que mejor encaja las posiciones de los jugadores (correlación + emparejamiento por distancias) y descartar repeticiones (sin homografía válida, o que no encajan con nada).
   - Código nuevo: `src/hackbarca/io/broadcast_align.py`. Reutiliza `load_match(source="broadcast")` (`src/hackbarca/io/load_match.py`, que ya admite `source`).
   - **Prueba con datos sintéticos**: partir un partido del banco en dos "cámaras" con cortes, repeticiones y desfases conocidos, y comprobar que el desfase se recupera exacto.
   - **Puerta de decisión** (agregado que ejecuta una persona): % de frames de hueco táctico que el broadcast alineado cubre con balón, y distancia mediana entre los dos balones cuando ambos lo ven. Seguimos si cubre ≥ 25 % de los huecos y la distancia es < 1,5 m.
3. **Detector de balón local en el vídeo táctico**, solo en los frames de hueco que queden sin cubrir.
   - Pesos abiertos entrenados en datos públicos de fútbol (tipo YOLO/SoccerNet); nada etiquetado a mano.
   - Detección por frame → proyección al campo con la homografía (`load_homography.py` del starter) → filtro temporal (Kalman, para descartar falsos positivos).
   - Código: `src/hackbarca/vision/ball_detector.py`. Se ejecuta una vez y deja un fichero de bolas detectadas en `data/derived/` (ignorado por git y versionado por hash).
   - **Puerta de decisión**: tasa de detección en los huecos y acuerdo con el ATD donde el ATD sí ve el balón (agregados).
4. **Balón fusionado**: ATD → broadcast → vídeo, con una columna que indique la fuente de cada posición. `frame_features` (`src/hackbarca/features/frame_features.py`) pasa a usarlo. En el banco se simula rellenando un % de huecos con el ruido medido para que el modelo aprenda a fiarse de cada fuente.
- **Reproducibilidad**: la entrega no puede depender de una GPU. Se versiona el fichero de bolas detectadas (es un derivado, no datos del reto) junto con el comando que lo regenera.
  - **Antes de versionarlo, confirmar con la organización que está permitido** (pregunta en issues), porque a fin de cuentas son posiciones sacadas del vídeo.

### Frente B: banco v3 y protocolo de selección (responsable: persona B)
1. **Descargar DFL a mano** (21 ficheros de figshare con el navegador) en `external_data/dfl/<id>/`. El cargador ya existe (`src/hackbarca/io/dfl.py`).
2. GT de DFL a partir de su **posesión y estado del balón por frame del tracking**, la misma naturaleza que la GT del reto. Conversor en `src/hackbarca/io/external.py` vía `from_kloppy`.
3. **Banco v3** = 7 DFL + 3 Metrica. Recalibrar `degrade` (`src/hackbarca/io/degrade.py`, modelo "long") con los mismos objetivos que v2 (8 números publicados + estructura de huecos de E005/E006) y añadir como objetivo la distribución de duraciones de posesión de la GT pública.
4. **Protocolo pre-registrado** (ADR nuevo): un cambio se acepta solo si mejora el spellF1 medio en 10 pliegues dejando un partido fuera por encima del ruido medido con bootstrap. Los 5 min públicos son solo una comprobación de cordura y nunca deciden.
5. Mantener `error_analysis` (`src/hackbarca/pipeline/error_analysis.py`) para medir cada iteración.

### Frente C: modelo de segmentación (responsable: persona C + agente)
1. **Features v2** en `frame_features.py`:
   - toques y pases: cambios bruscos de dirección o velocidad del balón, quién está más cerca en ese instante y hacia qué equipo va la trayectoria;
   - cinemática individual de los k jugadores más cercanos (velocidad hacia el balón, deceleración al recibir);
   - presión (rivales a < 5 m del poseedor) y compacidad de los bloques.
2. **Modelo secuencial bidireccional** (TCN o GRU, PyTorch en CPU/GPU) sobre ventanas de features. Se ensambla con LightGBM promediando probabilidades. Código: `src/hackbarca/models/seq.py`. Para que sea determinista, semillas fijas y `torch.use_deterministic_algorithms`.
3. **Decodificación HSMM** con prior de duración por estado (H/A/D) aprendido del banco, en `src/hackbarca/decode/hsmm.py`. Reutiliza la interfaz de `src/hackbarca/decode/viterbi.py`.
4. **Ajuste fino de fronteras**: llevar cada cambio de posesión al toque detectado más cercano dentro de ±0,5 s.
5. **Adaptación al partido real**: autoentrenamiento con pseudoetiquetas de alta confianza sobre el ATD real (sin ninguna GT) y ensamblado de semillas.

## Calendario
| semana | A (visión) | B (banco) | C (modelo) |
|---|---|---|---|
| 21-27 oct | descarga broadcast; alineación + prueba sintética; puerta de cobertura | DFL descargado; banco v3 calibrado; ADR del protocolo | features v2; HSMM |
| 28 oct-3 nov | fusión del broadcast en el pipeline; prototipo del detector | GT de DFL vs GT pública: comparar duraciones | modelo secuencial + ensamblado |
| 4-10 nov | detector en los huecos + Kalman; balón fusionado completo | evaluación de cada candidato en v3 | pseudoetiquetado; fronteras a los toques |
| 11-17 nov | congelar artefactos | prueba de reproducción en un clon limpio (otra persona) | documento técnico |

- **10 nov**: confirmar que los 3 estáis inscritos como "La MasIA" (cierre de inscripciones).
- **14 nov**: congelar el modelo.
- **15-16 nov**: subir `La MasIA.csv` y `La MasIA.pdf`, con el repo ya público.
- **Siempre** hay un CSV válido de reserva: el L3 actual y, por debajo, el baseline.

## Ganancia esperada (orientativa, en el banco)
- Features v2 + HSMM + fronteras ajustadas: +0,03 a +0,06.
- Modelo secuencial + ensamblado: +0,02 a +0,05.
- Banco v3 / DFL: más fiabilidad para elegir, y probablemente +0,02 a +0,04 por entrenar con 10 partidos.
- Balón fusionado: el mayor potencial. Si cubre la mitad de los huecos, el acierto en esos frames puede pasar de 0,74 a ~0,85, lo que daría +0,05 a +0,10. Es la apuesta para ganar.

## Riesgos
- **La alineación del broadcast no converge** → pasar directamente al detector de vídeo.
- **El detector falla con un balón tan pequeño** → usarlo solo con alta confianza, como feature y no como verdad.
- **El banco engaña** → v3 con GT basada en tracking, más el protocolo pre-registrado.
- **No reproducible** → artefactos versionados y prueba en un clon limpio antes del 14 nov.

## Verificación
- Cada pieza nueva tiene tests con datos sintéticos (`uv run pytest`).
- Cada cambio se mide en el banco v3, dejando un partido fuera, y se registra en `docs/experimentos.md`.
- Las piezas que tocan datos reales solo imprimen agregados y las ejecuta una persona.
- Antes de entregar: `make_submission --model <final> --team-name "La MasIA"` → validador oficial → sha256 igual en un clon limpio → comprobación con los 5 min públicos (`run_evaluation.py --gt public`).
