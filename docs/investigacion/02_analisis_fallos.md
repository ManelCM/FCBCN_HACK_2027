# Análisis a fondo de los fallos del modelo vigente (L3 apilado)

- **Fecha**: 2026-10-22 · **Autor**: Manel (con agente)
- **Modelo**: L3 apilado, features v3, sin Smart Tagging (E018, ADR-0011). Predicciones fuera de
  muestra (dejando un partido fuera) y decodificación de E019 (T 1, switch 20, to_dead 10, from_dead 10).
- **Datos**: banco v2.1 (Metrica g1-3 degradados). Solo datos externos.
- **Reproducir**: `uv run python -m hackbarca.pipeline.failure_analysis [--controlled 1.5]`
  (resumen en `external_data/failure_analysis*.json`).

## 0. Primer hallazgo: el banco medía una definición de posesión equivocada

`CHALLENGE.md` define el cambio de posesión como el momento en que el rival gana el balón
**controlado**. Según ese mismo texto, no cuentan:
- un toque sin control,
- un desvío de un pase que no lo intercepta,
- un duelo aéreo ganado sin control posterior.

Como pistas de control propone que el jugador conserve el balón unos segundos o que el equipo
encadene más de dos pases.

La GT de nuestro banco salía de los eventos de Metrica y cambiaba de equipo en **cada**
recuperación o pérdida. Con la regla de control (`external.controlled_possession`: un robo solo
cuenta si el nuevo equipo conserva el balón T segundos; tras un D el poseedor vale siempre):

| T | posesiones | media | mediana | < 3 s | spellF1 del modelo actual (sin reentrenar) |
|---|---|---|---|---|---|
| 0 (GT por eventos) | 1.138 | 10,4 s | 4,8 s | 35,9 % | 0,542 |
| 1 s | 956 | 12,4 s | 6,8 s | 24,2 % | 0,600 |
| **1,5 s** | 868 | 13,7 s | 8,4 s | 17,5 % | **0,628** |
| 2 s | 802 | 14,8 s | 9,5 s | 13,3 % | 0,640 |
| 3 s | 685 | 17,3 s | 11,9 s | 5,4 % | 0,661 |
| **GT pública del reto** | 17 en 5 min | **14,2 s** | — | **17,6 %** (3/17) | — |

Con T = 1,5 s el banco reproduce casi exactamente la estructura de la GT pública (13,7 s y 17,5 %, frente a 14,2 s y 17,6 %). **El 40 % de lo que
llamábamos «posesiones perdidas» eran robos de 1 s que en el reto no existen.**

Usamos la GT pública solo como **estadística descriptiva** (duraciones) para fijar la definición,
como prevé el plan y como pide `CHALLENGE.md` («if your reading of the definition disagrees with
it, the ground truth is right»). No elegimos ningún modelo por su spellF1. T = 1,5 s se adopta
provisionalmente y se comprobará la robustez con T = 1 y T = 2 (E021).

**El resto del análisis usa la GT controlada (T = 1,5 s).**

## 1. Nota y tipo de error

spellF1 **0,628**, con 738 posesiones predichas frente a 868 reales.

| IoU | precisión | recall |
|---|---|---|
| 0,3 | 0,80 | 0,68 |
| 0,5 | 0,71 | 0,60 |
| 0,7 | 0,52 | 0,45 |

La precisión es mayor que el recall en todos los umbrales: **nos faltan posesiones** (fusionamos
de más), más que inventarlas.

## 2. Por qué se pierde cada posesión (taxonomía)

Se pierden 344 de 868 (39,6 %) a IoU 0,5. Para cada posesión real sin pareja:

| causa | nº | % | duración mediana | qué significa |
|---|---|---|---|---|
| **equipo contrario** | 145 | 42 % | 3,3 s | En ese tramo predecimos al otro equipo: no vimos el robo |
| **fusionada: no vimos una posesión rival corta** | 79 | 23 % | 6,7 s | Se une con la siguiente del mismo equipo porque la del rival en medio desapareció |
| **fusionada: ambos** | 49 | 14 % | 4,9 s | Igual, con un parón también en medio |
| tapada por D | 33 | 10 % | 3,5 s | Predecimos balón muerto con el juego vivo |
| fusionada: parón no detectado | 9 | 3 % | 4,3 s | Faltó el D que la separaba de la siguiente |
| fronteras desplazadas o recortadas | 18 | 5 % | 4,1 s | Encontrada, pero con un solape de menos del 50 % |
| partida en trozos | 11 | 3 % | 36 s | Una posesión larga rota en varias |

**Mecanismo principal: un cambio de posesión no detectado cuesta 2-3 posesiones.** Si no vemos que
el rival roba el balón durante 4 s:
- perdemos esa posesión (equipo contrario);
- las del equipo original de antes y después se funden en una sola, que ya no empareja con
  ninguna de las dos (fusionada).

Esa es la causa del 79 % de las pérdidas.

Posesiones predichas sobrantes (falsos positivos):
- 122 solapan una real pero mal ajustadas (suelen ser las fusionadas);
- 67 son de un equipo que no tenía el balón;
- 25 son trozos cortos de una posesión real.

## 3. En qué contexto se falla

**Por duración.**

| duración | encontradas |
|---|---|
| < 3 s | 24 % |
| 3-10 s | 52 % |
| 10-30 s | 80 % |
| > 30 s | 87 % |

**Por huecos de balón.** Las posesiones con más de la mitad en un hueco largo se encuentran un
**37,5 %**, frente al 64 % sin hueco. Comparando a igual duración (GT por eventos), en 3-10 s
pasan de 57 % a 33 %, y en más de 30 s, de 96 % a 57 %.

**Por cómo empiezan.**

| inicio | encontradas | n |
|---|---|---|
| saque de centro | 89 % | 18 |
| falta | 80 % | 92 |
| saque de banda | 67 % | 110 |
| **intercepción** | **53 %** | 305 |
| **balón suelto** | **58 %** | 220 |
| **córner** | **42 %** | 26 |

Las reanudaciones tienen un poseedor claro. Los robos en juego, no.

**Por cómo acaban.**

| final | encontradas |
|---|---|
| tiro | 64-70 % |
| pase interceptado | 62 % |
| cabezazo | 44 % |
| **despeje** | **12,5 %** |

Los balones divididos, aéreos y despejes son el punto débil.

**Por número de pases propios.**

| pases | encontradas |
|---|---|
| 0 | 34 % |
| 1-2 | 54 % |
| 3-5 | 77 % |
| 6 o más | 91 % |

Las posesiones sin juego elaborado son las difíciles.

**Por zona.** Apenas importa (56-66 %). El último tercio es algo peor.

## 4. Fronteras

Para cada transición real, ¿hay una predicha del mismo tipo cerca?

| transición | n | ≤ 0,5 s | ≤ 1 s | ≤ 2 s | desfase mediano | falsas |
|---|---|---|---|---|---|---|
| cambio de equipo (H↔A) | 621 | 27 % | 37 % | **49 %** | +1 frame | 145 |
| parón (H/A→D) | 241 | 48 % | 53 % | 59 % | +1 frame | 145 |
| reanudación (D→H/A) | 241 | 43 % | 56 % | 73 % | 0 | 109 |

- **Sin sesgo de retraso**: el desfase mediano es de 0 a 1 frame.
- **La mitad de los cambios de equipo no tienen ninguna frontera predicha a menos de 2 s.** No
  llegamos tarde; directamente no los vemos.
- Los parones se detectan peor que las reanudaciones. Además hay 145 parones falsos, sobre todo en
  huecos largos.

## 5. Frame a frame

| contexto | % de frames | acierto | detalle |
|---|---|---|---|
| balón visible | 61 % | **0,916** | errores sobre todo H↔A (8 %) |
| hueco corto (≤ 4 s) | 1 % | 0,896 | — |
| **hueco largo (> 4 s)** | **38 %** | **0,778** | el juego vivo se predice D en el **22 %** y el otro equipo en el 12-13 % |

**Por fase de juego** (frames vivos):

| fase | acierto |
|---|---|
| balón en vuelo (pase) | 0,91 |
| resto | 0,83 |
| **± 0,5 s de un duelo** | **0,70** |

**Señal del jugador más cercano.** Con el balón degradado, el más cercano es del equipo poseedor
solo el **70 %** de las veces; con el balón limpio, el 79 %. Incluso con el balón perfecto, uno de
cada cinco frames la cercanía engaña (pases en vuelo, presión). El modelo ya corrige parte: acierta
el 83 % de los frames en los que el más cercano es del rival.

**Calibración** (probabilidad máxima frente al acierto):

| probabilidad | % de frames | acierto |
|---|---|---|
| ≥ 0,9 | 80 % | 0,92 |
| 0,7-0,9 | 11 % | 0,67 |
| 0,5-0,7 | 8 % | 0,52 |

Ligera sobreconfianza en la franja media.

## 6. Cuánto valdría arreglar cada cosa (oráculos)

spellF1 si la predicción fuera perfecta solo en estos frames (base 0,628):

| escenario | spellF1 | ganancia |
|---|---|---|
| equipo perfecto (frames vivos) | 0,942 | +0,314 |
| causa «equipo contrario» resuelta | 0,829 | +0,201 |
| **huecos largos perfectos (balón fusionado ideal)** | **0,786** | **+0,158** |
| causa «fusionada: posesión rival corta» resuelta | 0,752 | +0,124 |
| posesiones < 3 s perfectas | 0,727 | +0,099 |
| **la mitad de los huecos largos perfectos** | **0,709** | **+0,081** |
| causa «fusionada: ambos» resuelta | 0,709 | +0,081 |
| D perfecto (frames D) | 0,669 | +0,041 |
| causa «tapada por D» resuelta | 0,656 | +0,028 |
| partida en trozos / fronteras desplazadas | 0,64 | +0,01 a +0,015 |
| huecos cortos perfectos | 0,627 | 0 |

Las causas se solapan, así que estas ganancias no se suman. Un oráculo parcial también puede
romper posesiones; por ejemplo, «balón en vuelo perfecto» da −0,04.

## 7. Conclusiones: de las causas a las soluciones

| # | causa raíz | evidencia | solución propuesta | ganancia esperada |
|---|---|---|---|---|
| 1 | **La GT del banco no seguía la definición del reto** | 36 % frente a 18 % de posesiones < 3 s | Banco con posesión controlada; reentrenar y reajustar la decodificación (E021); ADR-0012 | Ya +0,086 sin reentrenar; falta medir el reentrenado |
| 2 | **No vemos la mitad de los robos** (cambios H↔A) | 49 % detectados a ≤ 2 s; un robo perdido cuesta 2-3 posesiones | Señales específicas de cambio de posesión: reacción colectiva (un bloque se abre y el otro se cierra, la velocidad de avance se invierte; v4 dio +0,009 con 3 partidos), cambio de sentido del balón respecto a los equipos, y modelo a nivel de segmento que decida si un tramo candidato es una posesión rival real | Alta: es el 79 % de las pérdidas |
| 3 | **Huecos largos**: 38 % de los frames con 0,78 de acierto; el juego vivo sale D el 22 % | Oráculo +0,158 (ideal) / +0,081 (mitad) | **Balón fusionado** (broadcast + vídeo); y dentro del hueco, detector de balón parado por la colocación (jugadores quietos, barrera, córner, portero con el balón, jugador en la banda) | +0,05 a +0,10 |
| 4 | Duelos, aéreos y despejes | 0,70 de acierto cerca de un duelo; despejes 12,5 % | Pocas opciones en 2D. Altura del balón desde el vídeo (tamaño, parábola), a explorar con el detector | Baja-media |
| 5 | Infrasegmentación general | 738 frente a 868; P > R | Reajustar Viterbi con la GT nueva; HSMM con prior de duración (mínimo ~1,5 s por la regla de control) | +0,01 a +0,03 |
| 6 | Sobreconfianza en 0,7-0,9 | calibración | Temperatura en la decodificación (ya se busca) | Pequeña |

**Lo que NO es el problema:**
- el retraso de las fronteras (desfase de 0 a 1 frame);
- los huecos cortos (oráculo 0);
- partir posesiones largas (+0,015).

**Pendiente**: repetir este análisis con DFL (10 partidos) y con el modelo reentrenado en la GT
controlada.
