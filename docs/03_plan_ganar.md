# Plan para ganar la fase 1 (La MasIA)

> Aprobado el 2026-10-21. Reparto: A = visión y fusión del balón, B = banco v3 y protocolo, C = modelo. El agente escribe el código de los tres frentes; las piezas que tocan datos reales las ejecuta una persona (ADR-0004).


## Contexto
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
