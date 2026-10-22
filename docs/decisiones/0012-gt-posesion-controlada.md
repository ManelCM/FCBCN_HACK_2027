# 0012 — GT del banco con posesión «controlada» (un robo cuenta si dura ≥ 1,5 s)

- **Estado**: Propuesta (pendiente de E021: reentrenar con la GT nueva)
- **Fecha**: 2026-10-22
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
La GT del banco (ADR-0006) sale de los eventos de Metrica y cambia de equipo en **cada**
recuperación o pérdida. Pero `CHALLENGE.md` define el cambio de posesión como el momento en que el
rival gana el balón **controlado**. Según ese texto, no cuentan:
- un toque sin control,
- un desvío que no intercepta,
- un duelo aéreo sin control posterior.

Como pistas de control propone que el jugador conserve el balón unos segundos o que el equipo dé
más de dos pases. Añade: «if your reading of the definition disagrees with [the public ground
truth], the ground truth is right».

El análisis de fallos (`docs/investigacion/02_analisis_fallos.md`) muestra la discrepancia:

| | posesiones < 3 s | duración media |
|---|---|---|
| Banco (GT por eventos) | 35,9 % | 10,4 s |
| GT pública del reto | 17,6 % (3/17) | 14,2 s |

Estábamos entrenando, ajustando la decodificación y eligiendo modelos para perseguir robos de 1 s
que el reto no considera posesiones.

## Opciones consideradas
1. **Mantener la GT por eventos**: optimiza otra definición; el 40 % de las «posesiones perdidas»
   no existen en el reto. Descartada.
2. **Regla de tiempo**: un robo solo cuenta si el nuevo equipo conserva el balón ≥ T s; tras un D el
   poseedor vale siempre, porque al reanudar hay un poseedor claro. Simple, sin parámetros
   ocultos, y T se fija con la estadística descriptiva de la GT pública.
3. **Regla de tiempo o pases** (≥ T s o ≥ 2 pases propios): más fiel al texto, pero añade un
   parámetro más y en el ATD real no vemos los pases.
4. **Esperar a DFL**: su GT sale del tracking corregido, como la del reto. Es la verificación
   definitiva, pero la descarga es manual y no sabemos cuándo estará.

## Decisión (propuesta)
Opción 2 con **T = 1,5 s** (`hackbarca.io.external.controlled_possession`, opción `--controlled
1.5` en `tune_l3` y en `failure_analysis`). Al llegar DFL, se contrasta su distribución de
duraciones con la del banco controlado.

## Justificación
| T | posesiones | media | < 3 s | spellF1 del modelo actual (sin reentrenar) |
|---|---|---|---|---|
| 0 | 1.138 | 10,4 s | 35,9 % | 0,542 |
| 1 s | 956 | 12,4 s | 24,2 % | 0,600 |
| **1,5 s** | **868** | **13,7 s** | **17,5 %** | **0,628** |
| 2 s | 802 | 14,8 s | 13,3 % | 0,640 |
| GT pública | 17 | 14,2 s | 17,6 % | — |

- T = 1,5 s reproduce la media y el % de posesiones cortas de la GT pública.
- No se usa el spellF1 de los 5 min públicos para nada (se respeta ADR-0008): solo la forma de su
  distribución de duraciones, que es lo que `CHALLENGE.md` dice que hay que mirar para entender la
  definición.
- Pendiente E021: reentrenar el apilado y reajustar Viterbi con la GT controlada, y comparar con el
  modelo actual re-decodificado.

## Consecuencias
- **Todo número anterior (E001-E019) se midió con la GT por eventos** y no se compara con los
  nuevos. El orden de los modelos debería mantenerse, pero se re-evalúa el vigente antes de
  aceptar cambios (ADR-0008).
- 17 posesiones son pocas: la media de 14,2 s tiene un error grande. Por eso se reporta también
  con T = 1 y T = 2, y un cambio solo se acepta si mejora con los tres valores.
- Se revisa cuando llegue DFL o si la organización aclara la definición.
