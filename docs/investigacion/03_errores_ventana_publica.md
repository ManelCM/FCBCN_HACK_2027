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

## Pendiente
- Repetir con el candidato de familias (`submissions/candidato_familias.csv`) y comparar los cinco errores.
