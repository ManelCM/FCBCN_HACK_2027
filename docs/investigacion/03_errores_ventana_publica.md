# Errores de la entrega en la ventana pública de 5 minutos, frame a frame

- **Fecha**: 2026-10-24 · **Autor**: Manel (con agente) · **Script**:
  `uv run python -m hackbarca.pipeline.public_window "La MasIA"`
- **Autorización**: Manel autorizó que el agente vea nuestras predicciones frame a frame **solo dentro
  de la ventana pública** (video frames 11.185-18.684), junto a la GT pública, para analizar errores. La
  GT de la ventana es pública y las predicciones son la salida de nuestro algoritmo. Los datos del reto en
  bruto siguen fuera del alcance del agente (T&C, ADR-0004). **Pendiente de que el equipo lo recoja en el
  ADR-0004.**
- **Modelo analizado**: la entrega actual (apilado v3, sha `252716b0…`). Tiempos desde el inicio de la
  ventana.
- **Cautela**: con 17 posesiones esto enseña **tipos de error**, no sirve para elegir modelos (ruido
  ±0,38).

## Cifras
- spellF1 0,556, macroF1 0,766, acierto por frame 0,830.
- 19 posesiones predichas frente a 17 reales; 11 emparejadas a IoU 0,5.
- Por frame, el equipo se acierta bien (94 % con el balón vivo en los dos). **El balón muerto, no**: de
  1.469 frames D, solo predecimos D en 594.

## Lo que hacemos bien
- Las posesiones largas: 41,6 s, 28,6 s, 42,4 s y 35,7 s, todas al 98-100 %.
- Varios cambios a menos de 1 s: el robo de 0:59 (+0,6 s), el parón de 1:06 (−0,1 s) y su reanudación
  (−0,4 s), y los cambios de 2:26 (+0,6 s) y 4:04 (+0,1 s).

## Los cinco errores
1. **Robos cortos que no vemos o vemos tarde.**
   - Recuperación del Barça a 0:06.6: la damos 3,5 s tarde.
   - Posesión del Athletic de 2,8 s (0:14.7-0:17.6): no la vemos y fusionamos dos posesiones del Barça.
   - Posesión del Athletic de 3,2 s (1:40.8): la vemos 1,2 s tarde y solo cubrimos el 63 %.
2. **Microposesiones de menos de 1,5 s** (H durante 0,8 s a 2:46.4 y durante 1,3 s a 3:03.2, dentro de una
   posesión del Athletic): no las vemos.
   - **Implicación para el banco**: la GT pública **sí cuenta posesiones de menos de 1,5 s**. La regla
     controlada (ADR-0012, T = 1,5 s) reproduce la distribución global (17,6 % de menos de 3 s), pero
     borraría estas microposesiones. Hay que revisar si conviene una regla distinta, por ejemplo con
     condición de pases o de distancia, o medir también con T = 1 s.
3. **Parón de 20 s (3:08-3:28) dado como juego vivo**: D solo el 19 %. Entramos en D 10 s tarde y salimos
   6 s antes.
4. **Parón de 18 s (4:07-4:26)**: D al 80 %, pero con un cambio falso H→A justo antes y 3 s de retraso.
5. **El peor tramo, 4:26-4:48**: posesión del Athletic de 7,6 s y parón de 14,5 s dados como juego vivo,
   con **parpadeo** A/H/A/H/A (5 cambios falsos). Probablemente es un balón parado con el balón visible y
   los jugadores casi quietos.

## Qué lo ataca
| patrón | qué debería corregirlo |
|---|---|
| Parones con el balón visible confundidos con juego vivo (3 de 4 parones) | Familias **reanudación** (jugadores quietos, colocación) y **bloques** del candidato (ADR-0015) |
| Robos cortos (< 3 s) que no vemos o vemos tarde | Familias **jugadores** (presión, sprints) y **balón** (sentido de progresión) |
| Microposesiones (< 1,5 s) | Revisar la regla controlada del banco |

## Repetición con la entrega E070 (25 oct, sha `9602959c4ec05bd6`)
- spellF1 0,651, macroF1 0,803, acierto por frame 0,842 (antes 0,830).
- **30 segmentos predichos frente a 21 reales**. De los 11 cambios sobrantes, **todos pasan por D**, y la
  mayoría son D de 0,3-4 s que parten una posesión viva en dos: 0:17, 1:43, 1:54, 2:41, 3:18, 3:22, 3:24,
  4:11, 4:41 y 4:43.
- **Comparación con los cinco errores del 24 oct**:

  | error | antes (`252716b0…`) | ahora (E070) |
  |---|---|---|
  | 1. Robos cortos tarde o no vistos | recuperación de 0:06 a +3,5 s; posesión A de 2,8 s fusionada | 0:06 a **+4,5 s**; la A de 2,8 s **sí se ve** (−0,2 s); el robo de 0:59 llega a +1,6 s |
  | 2. Microposesiones < 1,5 s | no se ven | siguen sin verse (2:46 y 3:03); la H de 3,5 s a 4:04 tampoco |
  | 3. Parón de 20 s (3:08-3:28) | D al 19 % | D al 33 %, con 4 cambios H/D sobrantes dentro |
  | 4. Parón de 18 s (4:07-4:26) | D al 80 %, cambio falso antes | D al 69 %; entramos 31 s tarde porque no vemos la H de 4:04 |
  | 5. Tramo 4:26-4:48 con parpadeo | 5 cambios falsos | D al 53 %; 4 cambios H/D sobrantes |

- **Lectura**: el parpadeo entre equipos ha desaparecido, pero ahora el error dominante son **D cortos sobrantes
  con el balón visible**. En el partido completo, `real_vs_bench` (E041) mide 0,68 tramos D de menos de 5 s por
  minuto en el reto, frente a 0,09 en la GT del banco: es un patrón del partido entero, no de la ventana. Se
  analiza en `04_analisis_fallos_v3.md` §6.

## Pendiente
- ~~Repetir con el candidato de familias y comparar los cinco errores.~~ Hecho arriba.
