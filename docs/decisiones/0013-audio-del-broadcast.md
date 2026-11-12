# 0013 — Audio del broadcast (silbato y comentarios) como señal experimental

- **Estado**: Propuesta. **Desbloqueada (22 oct)**: `broadcast.mp4` tiene audio real de televisión
  (−26 dB de media); el del táctico está mudo (−60 dB). La grabación del broadcast es continua en
  cada parte (E026: el 99 % de los frames emparejados a ±2 frames de un único desfase), así que
  todo el audio se lleva al timeline táctico con un desfase por parte. Falta saber si hay
  comentarista (que lo escuche una persona) y la respuesta a Q6.
- **Fecha**: 2026-10-22
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
Ninguna fuente visual (ATD, broadcast, vídeo táctico) da certeza en los huecos largos sin balón:
el 38-41 % de los frames, con un acierto de 0,78 (`docs/investigacion/02_analisis_fallos.md`).
Surge la idea de usar los comentarios del partido y transcribirlos.

Lo que dicen las bases:
- `RULES.md` permite cualquier método inspeccionable y mejorar o sustituir los datos dados
  («Improving that data, or replacing it with your own, is entirely legitimate»).
- La regla 2 prohíbe «any other source of corrected possession data for this fixture».
- La regla 3 exige regenerar el CSV con los datos entregados.

## Opciones consideradas
1. **Audio del `broadcast.mp4` entregado**: permitido. Las etiquetas siguen saliendo de un
   algoritmo.
2. **Comentarios externos de este partido** (crónica minuto a minuto, radio, YouTube): cerca de la
   regla 2, y la organización no podría regenerar el CSV solo con los datos entregados.
   **Descartada.**
3. **Transcripción con un servicio en la nube**: sería subir datos del reto (ADR-0004).
   **Descartada.** Todo se procesa en local.

## Decisión (propuesta)
Explorar la opción 1 por fases, **detrás** de E021 y del balón fusionado:

1. **Comprobar si hay audio**, lo hace una persona: `ffprobe -v error -show_entries
   stream=codec_type,channels -of csv data/broadcast/BAR-ATH_broadcast.mp4`. La persona escucha si
   hay comentarios o solo ambiente. El agente no recibe audio ni transcripción.
2. **Detector de silbato** (procesado de señal, banda de 2-4 kHz): precisión de frame, marca el
   inicio de muchos D (faltas, fueras de juego, goles, finales de parte) y sirve en cualquier
   idioma. No cubre los saques de banda ni de puerta.
3. **Comentarios** (solo si 2 funciona): transcripción local con faster-whisper (15-20 min en la
   RTX 3060). Nombres cruzados con las plantillas públicas para sacar el equipo, y palabras clave
   para el balón parado. Útil solo dentro de los huecos largos, porque el comentarista va 1-3 s
   tarde y 1 s de retraso cuesta ~35 % de spellF1.

## Justificación
- Ataca los dos errores medidos: el D dentro de los huecos (el juego vivo sale D el 22 % de las
  veces) y el equipo dentro de los huecos.
- Riesgo de validación: los partidos de Metrica no tienen audio, así que **no se puede medir en el
  banco**. Alternativas:
  - coherencia con señales propias (tras un silbato, ¿baja la velocidad colectiva? ¿coincide con
    los SET PIECE del Smart Tagging?), solo con agregados;
  - SoccerNet (partidos de tele con audio y marcas de «balón fuera», «falta», «saque»; requiere
    firmar un NDA).

## Consecuencias
- Requiere la alineación broadcast → táctico (ADR-0010), que espera a `broadcast atd.txt`.
- Si la grabación es continua, el desfase es fijo por mitad y el audio sigue en directo durante las
  repeticiones.
- Pregunta Q6 a la organización para confirmar.
- Si no hay audio, se archiva.
