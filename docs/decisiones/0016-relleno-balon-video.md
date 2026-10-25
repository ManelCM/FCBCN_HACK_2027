# 0016 — Relleno del balón en los huecos del ATD con un YOLO afinado con pseudoetiquetas

- **Estado**: **Rechazada** (25 oct). E068: en la ventana pública los parones salen como juego vivo. E069: con una simulación fiel, +0,008 en 6/10 partidos, por debajo del protocolo
- **Fecha**: 2026-10-25
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
El ATD no tiene balón en el ~41 % de los frames evaluados, sobre todo en huecos largos (mediana 268 frames).
Ahí el modelo confunde juego vivo con balón muerto (acierto 0,78 en huecos largos, E020) y en la ventana
pública 3 de 4 parones salen como juego vivo. ADR-0010 buscaba otra fuente de balón:
- el broadcast solo rellenaba el 7,5 % de los huecos (E023);
- el YOLO genérico (COCO) solo dejaba el 24 % a < 2 m en huecos simulados (E025).

## Opciones consideradas
1. **Sin relleno** (modelo E035, 0,7185 en el banco v3).
2. **YOLO genérico**: descartado en E025.
3. **YOLO afinado con pseudoetiquetas del propio ATD**: el balón del ATD proyectado con la homografía oficial
   etiqueta los recortes, sin ninguna etiqueta manual (RULES.md). Elegida.
4. **Inferir el balón con un modelo de jugadores** (tipo Ball Radar): más caro, sin probar.

## Decisión
- `vision/pseudo_labels.py` construye el conjunto (5.974 recortes con balón y 1.456 sin balón, excluidos
  ±10 s alrededor de los tramos de prueba) y afina YOLOv8n durante 30 épocas.
- `vision/ball_detector.py --class-id 0 --conf 0.01 --conf-ref auto` detecta y enlaza en los huecos reales y
  escribe `data/derived/video_ball.npz`. `conf_ref` se saca del propio detector, sin GT.
- `io/video_fill.py`:
  - **reto**: el balón del vídeo entra solo en los frames sin balón del ATD;
  - **banco**: se simula el mismo relleno con la forma medida en el reto, para que entrenamiento y reto vean
    el mismo dominio.
- El artefacto lo activa con `video_fill` en `l3_decode.json` (`train_l3_final --video-fill`) y
  `make_submission` lo aplica.

## Justificación
- **E061**: en huecos simulados, el 59,6 % de los frames queda a < 2 m tras enlazar (puerta ≥ 50 %); el genérico
  dejaba el 24 %.
- **E063**: cubre el **55,9 %** de los huecos reales. El balón relleno va a 0,64 m/s de mediana: casi todo es
  balón parado en parones y reanudaciones, justo donde el modelo fallaba.
- **Desajuste reto/banco**: con el relleno simulado v2 baja de 0,060 a **0,045** (columnas ≥ 0,25: de 6 a 4).
- **E064**: **0,7484** frente a 0,7185 (+0,030), con mejora en los **10 de 10** partidos.

## Riesgos y límites
- La ganancia del banco depende de que la simulación del relleno sea fiel. La cobertura y la velocidad se
  midieron en el reto; la precisión solo se mide en huecos simulados con juego vivo. En los parones reales
  no hay referencia: un falso positivo quieto (punto de penalti, un objeto de la banda) parecería balón
  parado.
- **Reproducibilidad**: la entrega depende de los pesos afinados (`best.pt`) y de `video_ball.npz`. Los dos se
  regeneran con los comandos de arriba (GPU recomendada: entrenar ~1 h, detectar ~3 h). Hay que comprobar
  que dos ejecuciones dan el mismo `video_ball.npz`, o versionar el `.npz` en el paquete de entrega si la
  organización lo permite (no contiene datos del reto, pero se deriva de ellos: preguntar).
- Comprobación antes de aceptar: CSV con y sin relleno en la ventana pública de 5 min. Sirve solo para
  detectar roturas, porque la banda de ruido es grande (E022).

## E068 (25 oct): comprobación en el reto
- Ventana pública: spellF1 0,607 y macroF1 0,625 con relleno, frente a 0,651 y 0,803 sin relleno. 3 de los 4
  parones salen como juego vivo.
- Causa probable, vista en agregados: en el reto el balón relleno está pegado a un jugador (en parones, p50 1,9 m)
  y casi todo cae en parones. En el banco simulado estaba más lejos (3,1 m) y rellenaba demasiado juego vivo.
- Antes de adoptarlo hay que simular el relleno condicionado a la velocidad del balón, con el balón junto a un
  jugador en los parones, y repetir E064 y la ventana pública.

## E069 (25 oct): simulación fiel y decisión
- Relleno v3, calibrado con los agregados del reto: en juego vivo solo balón lento y junto a alguien; en
  parones, a veces llevado por quien saca; enganches a otro jugador.
- Resultado: 0,7262 frente a 0,7185 (+0,008), mejora en 6 de 10 partidos. No pasa ADR-0008.
- **Rechazada**: la entrega va sin relleno. El código (`io/video_fill.py`, `vision/*`) se conserva.
  Posible continuación: rellenar solo cuando un clasificador indique juego vivo.
