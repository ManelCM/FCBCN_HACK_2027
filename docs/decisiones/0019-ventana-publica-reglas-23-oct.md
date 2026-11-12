# 0019 — Ventana pública y etiquetas tras las reglas del 22-23 oct

- **Estado**: Propuesta
- **Fecha**: 2026-10-26
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
El repo oficial tiene dos commits nuevos (`c073f29`, 22 oct, y `f8f2629`, 23 oct). Solo cambian
documentos: RULES, FAQ, CHALLENGE, DATA, EVALUATION, OVERVIEW, README y SUBMISSION. El código de
evaluación no cambia. Lo que antes era consejo ahora es regla:
- **Ventana pública de 5 min**: «for checking that your pipeline is correct — frames aligned, file
  well-formed, labels meaning what you think. They are **not training data and not a tuning set**».
  Nunca sirve para elegir entre dos versiones (RULES.md §2, FAQ, EVALUATION).
- **Cero etiquetas humanas sobre BAR-ATH**, «in any amount and for any purpose», ni para entrenar ni
  para validar. Tampoco vale corregir el tracking a mano, ni convertir lo que se ve en el vídeo en
  «rules and constants tied to specific frames of the game».
- **Sí se permite**:
  - aprendizaje sin etiquetas humanas, incluidas las pseudoetiquetas que produce un algoritmo
    («provided no person decided any of them»);
  - mirar el vídeo y los datos para entender y diseñar.
- **Datos externos**: solo de otros partidos. Nada producido por personas sobre BAR-ATH (eventos,
  tracking corregido, anotaciones).
- **Etiquetas a mano sobre otros partidos**: permitidas en cualquier cantidad. El documento técnico
  tiene que listar cada conjunto etiquetado: fuente, quién lo etiquetó y cuánto.

Nuestro submódulo estaba en `3de6b60` (15 oct), con las reglas anteriores.

## Opciones consideradas
1. **No cambiar nada**: lo hecho fue conforme a las reglas de su fecha. Pero la entrega se juzga con las
   reglas vigentes y el documento técnico tiene que poder defenderlo todo. Descartada.
2. **Regla interna explícita y revisión de lo hecho**: se fija qué uso de la ventana pública queda y se
   revisa, decisión por decisión, lo que la usó. Elegida.

## Decisión
1. **Submódulo** en `f8f2629`.
2. **Ventana pública**: solo como prueba de humo:
   - el CSV tiene el formato bueno y los frames están alineados;
   - las etiquetas significan lo que creemos (H/A/D/X).

   **Nunca**:
   - como puerta de una señal o de una familia;
   - para calibrar un simulador o un parámetro;
   - para fijar un umbral;
   - para elegir entre versiones;
   - como dato de entrenamiento.

   Su spellF1 se informa en el LOG como humo, sin decidir nada con él.
3. **Puertas sobre el reto** (señales de vídeo y audio, guardarraíl de ADR-0018): solo **agregados** del ATD
   y **pseudoetiquetas algorítmicas**, es decir, nuestra predicción con pmax ≥ 0,9. Ninguna persona decide
   ninguna.
4. **Cero etiquetas a mano sobre BAR-ATH**. Toda etiqueta a mano de otros partidos se apunta en el registro
   de conjuntos etiquetados del documento técnico: qué, cómo, quién y cuánto.
5. **Arranque de la 2.ª parte** (FAQ: los frames anteriores al saque puntúan y son `D`):
   - no se fija ninguna constante ligada a frames concretos del partido;
   - se mide con un agregado qué predice el modelo en ese tramo;
   - si hiciera falta, se usaría una **regla algorítmica general**, por ejemplo «antes del primer
     movimiento del balón tras el inicio de una parte, D», que vale para cualquier partido.

## Revisión de lo hecho con la ventana pública
| decisión | qué usó | situación con la regla nueva | acción |
|---|---|---|---|
| ADR-0012 (T = 1,5 s) | forma de la distribución de duraciones de la GT pública (% de posesiones < 3 s), no su spellF1 | Es lo que la regla llama «labels meaning what you think»: entender la definición de posesión controlada. Pero T es un parámetro de evaluación fijado con un estadístico de la ventana | **Justificar T sin la ventana**. Ojo: ADR-0014 eligió T = 2 s en DFL con el mismo criterio, así que tampoco es independiente. Las vías son: (a) el texto de CHALLENGE.md («a player keeps the ball for a few seconds», es decir, T del orden de 1-3 s); (b) un experimento de robustez: que el orden de los candidatos con ADR-0008 sea el mismo con T = 1, 1,5, 2 y 3 s, en cuyo caso T no decide nada y basta con fijarlo por el texto. Si el orden cambia con T, decide el equipo y se documenta |
| E055-E056 (U1, plano de la tele) | AUC sobre la GT pública como segundo criterio | La puerta también pasa solo con la predicción segura (0,80; 0,78 en los huecos) y el simulador se calibró con la predicción segura | Quitar `auc_publica` del criterio en `real_signals` y `broadcast_shot_report` (puede quedar como humo, sin decidir) |
| ADR-0016 / E068 (relleno con YOLO) | comprobación en la ventana pública antes de aceptar | El rechazo formal sale del banco (E069: +0,008 en 6/10, por debajo de ADR-0008) | Redactar la consecuencia así en el ADR |
| ADR-0017 / M1.1 (banco v3.1) | 1,86 tramos lentos/min de la GT pública, como contexto | La calibración apunta al banco limpio (1,08) y a los agregados del ATD; ningún parámetro sale de la GT pública | Dejarlo escrito en ADR-0017 |
| E006 / banco v2 (`bench_config.BENCH_DEGRADE_V2`) | uno de los objetivos de la rejilla fue «56 % de D dentro de los huecos largos (GT pública)», junto con los 8 números publicados del baseline y los agregados del ATD | Un parámetro del simulador (sobre todo `dead_weight`) se calibró en parte con la GT pública | En la próxima recalibración (v3.2, L1), sustituir ese objetivo por agregados del ATD y pseudoetiquetas, y documentar el cambio |
| ADR-0018 (guardarraíl) | umbrales del banco y agregados de nuestra predicción | Conforme | — |

## Justificación
- La regla nueva es clara y la entrega se juzga con ella. Un documento técnico que lo explique
  decisión por decisión es la mejor defensa.
- La ventana pública es demasiado ruidosa para decidir (17 posesiones, ±0,38; ADR-0018). No perdemos
  información útil.
- Las pseudoetiquetas están permitidas de forma explícita, así que las puertas sobre el reto siguen
  siendo posibles.

## Consecuencias
- `docs/06_plan_parones.md` (L0-L3) y todo experimento nuevo siguen esta regla.
- Hay que corregir el código de las puertas que imprime cifras de la ventana pública como criterio.
- El documento técnico lleva:
  - una sección «uso de la ventana pública» con esta tabla;
  - el registro de conjuntos etiquetados.
