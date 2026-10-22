# 0010 — Fusión del balón: tracking broadcast alineado + detector en vídeo, y cómo se reproduce

- **Estado**: Propuesta. **Las dos fuentes, tal como están, no pasan su puerta (22 oct)**:
  broadcast alineado → 7,5 % de los huecos (E023); detector YOLO COCO → 24 % a < 2 m incluso en
  tramos donde el ATD ve el balón (E025). Queda abierta una variante sin probar: detector
  reentrenado con pseudoetiquetas del propio ATD.
- **Fecha**: 2026-10-21
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
En el 41 % de los frames evaluados falta el balón (134 huecos largos), y ahí el acierto por frame
de L3 cae de 0,89 a 0,74 (E009). Las bases permiten usar cualquier activo, también el vídeo y el
tracking broadcast (DATA.md: "Everything is fair game"), siempre que se procese **en local**
(ADR-0004) y que nada se etiquete a mano (RULES.md).

## Opciones consideradas
1. **No rellenar**: el modelo infiere el balón a partir de los jugadores. Es lo que hace hoy.
2. **Tracking broadcast alineado**: otra cámara con su propio tracker ve el balón en el 50,5 % de
   sus frames. Barato: no hay que entrenar nada, solo alinear timelines.
3. **Detector de balón en el vídeo táctico**: YOLO preentrenado en local con la GPU (RTX 3060).
   Más caro y con un balón pequeño en el plano general.
4. **2 + 3 en cascada**: ATD → broadcast → vídeo. Elegida (decisión del equipo, 21 oct).

## Decisión
- `hackbarca.io.broadcast_align`: cada plano del broadcast se sitúa en el timeline táctico
  buscando el desfase que mejor encaja a sus jugadores con los del mismo equipo. Se descartan la
  cámara lenta (prueba de deriva entre el primer y el último tercio del plano) y los primeros
  planos. En sintético: error de ±1 frame y > 80 % de los frames en directo recuperados.
  Puerta de decisión: `broadcast_report` (agregados). Seguimos si cubre ≥ 25 % de los huecos con
  una distancia mediana < 1,5 m al balón del ATD.
- `hackbarca.vision.ball_detector`: YOLO (COCO "sports ball") por mosaicos de 640 px solo en los
  frames de hueco, proyectado con la homografía oficial, más un enlace temporal Viterbi
  (`ball_track`) que elimina los falsos positivos. Dependencias en el grupo opcional `vision`
  (torch CUDA), que la entrega no necesita.
- Fusión: columna con la fuente del balón (0 = ATD, 1 = broadcast, 2 = vídeo). El modelo se
  entrena en el banco con huecos rellenados al ritmo y con el error medidos.
- **Reproducibilidad**: la correspondencia del broadcast se recalcula en CPU en ~1-2 min, así que
  entra en el comando de entrega. Las detecciones del vídeo necesitan GPU: se generan una vez y se
  versiona el fichero derivado (posiciones x, y por frame), junto con el comando que lo regenera.

## Riesgos y pregunta pendiente
- **Q4 a la organización**: ¿se puede versionar en el repo público un fichero derivado del vídeo
  (posiciones del balón detectadas por nuestro modelo)? Si no, la entrega tendrá que ejecutar el
  detector, con GPU o en CPU (lento).
- Si la alineación no converge, se pasa directamente al detector de vídeo.
- Si el detector falla con un balón tan pequeño, se usa solo con alta confianza y como feature,
  nunca como verdad.

## Consecuencias
- Hace falta descargar el `atd.txt` del broadcast (parte `-009` de Drive).
- El documento técnico tendrá que explicar la fusión y los pesos preentrenados usados.
