# 0023 — Aclaración de la organización del 3 nov: nada entrenado, ajustado ni calibrado con el vídeo o la GT del reto

- **Estado**: Aceptada por Manel el 10 nov (falta la revisión de las otras dos personas)
- **Fecha**: 2026-11-10
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐
- **Modifica**: ADR-0020 y la entrega de E075d/E090c (U6). Complementa ADR-0019.

## Contexto
El 3 nov la organización envió un email a todos los participantes. Dice que **manda sobre la documentación,
los emails y las respuestas anteriores**. No está en el repo oficial, que no tiene issues y cuyo último commit
es `f8f2629`, del 23 oct.

Lo encontramos el 10 nov en los repos públicos de otros equipos (`docs/investigacion/07_competidores.md` §0):
- `SAM-Analytics-2027/.../rules/CLARIFICATIONS.md`;
- `nilaymastaadmi/.../work/rules-clarification-2026-11-04.md`;
- `verneylmavt/.../ORGANIZER-CLARIFICATION.md`.

Las tres copias coinciden. **Falta confirmarlo con el email original**, que debería tener Manel.

**No se permite:**
- comparar la salida con el vídeo;
- etiquetas a mano u otros datos creados manualmente sobre el partido del reto;
- subir los datos del reto a servicios de IA externos;
- **entrenar, ajustar o calibrar el algoritmo con la GT de posesión o con el vídeo del reto**.

**Se permite:**
- comprobaciones automáticas con los datos del reto;
- mirar los datos y el vídeo para entender qué es posesión;
- **usar como entrada** el vídeo, el tracking, el Smart Tagging u otros datos generados, si el algoritmo no se
  entrenó con la GT ni con el vídeo;
- ficheros de pasos intermedios;
- cualquier dataset con derechos de uso.

**Entrenamiento y ajuste:** solo con datos externos.

**Qué no cumple en nuestro proyecto:**
- La entrega vigente (E075 + regla de 3 s, sha `d976e575374b00d5`) usa U6, el conductor por vídeo.
- U6 sale de `vision/video_pass.py`, que carga `data/derived/ball_yolo/runs/atd_pl/weights/best.pt`. Ese
  YOLO está **afinado con recortes del vídeo táctico del reto** y pseudoetiquetas del ATD (E061).
- Eso es entrenar con el vídeo del reto. Las pseudoetiquetas no lo convierten en dato externo.
- Reentrenarlo igual (`weights_check`, `d72f781`) tampoco cumple. La otra sesión lo abandonó el 10 nov.

## Opciones consideradas
1. **Seguir con E075 hasta tener el email original**: arriesga la descalificación y obliga a medir todo lo
   nuevo dos veces.
2. **Rehacer U6 con un detector de balón entrenado solo con vídeo externo** (datasets de balón de SoccerNet),
   con la misma pasada: cumple, pero son días de GPU más una puerta y una evaluación nuevas, sin garantía de
   llegar al 14 nov.
3. **Volver a una entrega sin vídeo entrenado.** La mejor medida es **E058b** (cinco familias + F3
   `transition`): 0,7410 frente a 0,7254 de E049, mejor en 9 de 10. Pasó ADR-0008 y nunca se llevó a
   artefacto, porque E075 la superó.

## Decisión
- **Opción 3.** La entrega pasa al artefacto de E058b (E110), con la regla de D cortos de 3 s (ADR-0021) y la
  regla del saque (ADR-0022).
- Todo experimento nuevo se compara con E058b, sin `--video-conductor`.
- Queda abierta la opción 2 como línea propia si sobra tiempo.

## Justificación
- El texto prohíbe de forma explícita entrenar con el vídeo del reto, y U6 depende de ello.
- E058b es la mejor medida que no lo necesita (ADR-0008, `docs/experimentos.md`).

## Zonas grises que se documentan en el documento técnico
| pieza | qué usa del reto | lectura |
|---|---|---|
| simulador del banco (E006, E013, E047; ADR-0006, 0009, 0017) | agregados del ATD (p. ej. 41 % sin balón, dinámica del balón) para dar forma a la degradación de los partidos **externos** | no es la GT ni el vídeo, y no se ajusta ningún parámetro del modelo con el reto; otros equipos hacen lo mismo y lo declaran. Se explica |
| E006, `dead_weight` del banco v2 | uno de los objetivos fue el «56 % de D en los huecos largos» de la **GT pública** (ya señalado en ADR-0019) | con esta aclaración, ese objetivo era calibrar con la GT. **E119 lo justifica sin ella**: con un objetivo que solo sale del tracking (velocidad de los jugadores en los huecos frente a con balón) se elige 100 o más, y de 100 a 1.000 el banco apenas cambia. El banco v3.1 tampoco reproduce ya el 56 % (da 0,716). Se declara en el documento técnico |
| features relativas al partido (umbral de toque por partido, bloques en z-score por parte) | se calculan sobre el propio ATD en inferencia | cálculo fijo en inferencia, no ajuste |
| guardarraíles ADR-0018/0020 | agregados de nuestra predicción en el reto | comprobación automática de paso/fallo; no ajusta parámetros |
| k de la regla de D cortos (ADR-0021) | elegido solo con el banco | conforme |
| U10, T-DEED de SoccerNet sobre el broadcast | modelo externo sin reentrenar, con el broadcast como entrada | conforme como entrada; responde a Q6. No está en la entrega |
| agente que ve agregados (ADR-0004) | métricas agregadas | el texto prohíbe subir los datos; los agregados siguen pendientes de Q2 |

## Consecuencias
- E075, E075b, E075s, E098, E098b, E099 y todo lo que lleve `--video-conductor` quedan como historia. No
  entran en la entrega.
- El artefacto de E075 se conserva en `external_data/artefactos/e075/`.
- El documento técnico quita la variante A (U6) y añade una sección de cumplimiento con esta tabla.
- Se revisa al recibir el email original o una respuesta de la organización a Q2.
- **10 nov: E110 es la entrega provisional** (`La MasIA.csv`, sha `df01cd02b266e933`): el artefacto de E058b
  con la regla de 3 s y la del saque. Se reproduce igual en dos ejecuciones y el artefacto queda en
  `external_data/artefactos/e058b/`. En el reto, frente a las entregas anteriores sin vídeo, mejora las
  cuatro medidas del guardarraíl (E110 en `docs/experimentos.md`).
- **12 nov: la entrega pasa a E122** (E113 + poseedor al 30 %, sha `2223250da5edc7ed`) con el criterio de
  ADR-0024. Sigue sin vídeo ni GT del reto: el poseedor se entrena solo con el banco externo.
