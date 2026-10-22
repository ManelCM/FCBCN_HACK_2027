# Análisis a fondo de los fallos del modelo (L3 apilado), explicado

- **Fecha**: 2026-10-22 · **Autor**: Manel (con agente) · **Experimento**: E020
- **Modelo analizado**: L3 apilado, features v3, sin Smart Tagging (E018, ADR-0011).
- **Datos**: banco v2.1 = 3 partidos públicos de Metrica estropeados para parecerse al ATD. **Solo
  datos externos**: ningún dato del reto se ha mirado (ADR-0004).
- **Reproducir**: `uv run python -m hackbarca.pipeline.failure_analysis` (GT por eventos) y
  `... --controlled 1.5` (GT controlada). Resultados en bruto: `docs/investigacion/resultados/`.
- Si llegas nuevo, lee antes `docs/01_contexto_completo.md` (§1 explica spellF1).

---

## Resumen en un minuto

1. **El banco medía una definición de posesión equivocada.** El reto solo cuenta la posesión
   **controlada**: un toque, un desvío o un duelo sin control no cambian de equipo. Nuestro banco
   cambiaba de equipo en cada robo, aunque durase un segundo. Con la regla «un robo cuenta si el
   nuevo equipo conserva el balón ≥ 1,5 s», el banco se parece a la GT pública del reto (13,7 s de
   media y 17,5 % de posesiones < 3 s, frente a 14,2 s y 17,6 %), y **el mismo modelo, sin
   tocarlo, pasa de 0,542 a 0,628**. → ADR-0012. **Reentrenarlo con la nueva definición no
   mejora** (E021: 0,6225 frente a 0,6245): el cambio es de **medida**. El modelo era mejor de lo
   que creíamos; lo que estaba mal era el examen.
2. **El fallo real número uno: no vemos la mitad de los robos.** Solo el 49 % de los cambios de
   equipo tienen una frontera nuestra a menos de 2 s. Cada robo que no vemos cuesta 2-3
   posesiones. Es el **79 % de lo que perdemos**.
3. **El número dos: los huecos largos sin balón** (38 % de los frames). Ahí acertamos el 78 % de
   los frames, frente al 92 % con balón, y el juego vivo se confunde con balón muerto el 22 % de
   las veces. Rellenar la mitad de los huecos valdría +0,08.
4. **Después**: duelos, aéreos y despejes; y que damos menos posesiones de las que hay.
5. **No son el problema**: el retraso de las fronteras, los huecos cortos y partir posesiones
   largas.

---

## 1. Cómo se ha hecho el análisis

### Qué predicciones se analizan
- Para cada partido del banco se usan predicciones **fuera de muestra**: hechas por un modelo
  entrenado con los otros dos partidos. Son el equivalente honesto de lo que pasará con el
  Barça–Athletic, que el modelo no ha visto. Fichero: `external_data/oof_l3_notags_f3_stack.npz`,
  generado por `tune_l3 --no-tags --features 3 --stack`.
- Se decodifican con el Viterbi elegido en E019 (temperatura 1, cambio de equipo 20, entrar o
  salir de D 10).
- Sobre esas predicciones se compara con la verdad de dos formas: la GT por eventos (la de
  siempre) y la GT controlada (§2).

### Qué es una «posesión perdida»
spellF1 empareja cada posesión real con una predicha del mismo equipo si se solapan lo suficiente.
El solape (IoU) es la intersección dividida por la unión. Usamos el umbral **0,5**, el titular de
la métrica; la nota oficial promedia de 0,3 a 0,7. Una posesión real sin pareja es una **posesión
perdida** (un falso negativo). Una predicha sin pareja es una **sobrante** (un falso positivo).

### Cómo se clasifica cada posesión perdida (la taxonomía)
Para cada posesión real perdida se mira qué predijimos en sus frames y con qué posesión predicha se
solapa más. Las reglas se aplican en este orden (`failure_analysis.classify_spells`):

| causa | regla exacta | en palabras |
|---|---|---|
| consumida por otra | Hay una predicha con IoU ≥ 0,5, pero el emparejamiento ya la asignó a otra real | Casi nunca pasa (0 casos) |
| **equipo contrario** | ≥ 50 % de sus frames predichos del otro equipo | No vimos esta posesión: le dimos el balón al rival |
| **tapada por D** | ≥ 50 % de sus frames predichos D | Creímos que el balón estaba muerto con el juego vivo |
| **fusionada** | La predicha que más la cubre cubre ≥ 70 % de ella, pero es más larga | Acertamos el equipo, pero nuestra posesión se juntó con la siguiente o la anterior. Se subdivide según lo que había en medio en la realidad: un **parón** no detectado, una **posesión rival corta** no detectada, o **ambos** |
| **partida en trozos** | Dos o más predichas del mismo equipo la cubren y ninguna llega a IoU 0,5 | La rompimos en varias |
| **fronteras desplazadas o recortadas** | Resto: una predicha parecida pero mal ajustada | La encontramos, pero el solape no llega al 50 % |

### Qué es un «oráculo»
Es un experimento mental medible: **sustituimos nuestra predicción por la verdad solo en ciertos
frames** (por ejemplo, los huecos sin balón) y recalculamos spellF1. Dice cuánto ganaríamos si
resolviéramos perfectamente ese problema, que es el techo de una solución.

Dos advertencias:
- Varios oráculos atacan lo mismo, así que **sus ganancias no se suman**.
- Corregir solo una parte de los frames puede romper posesiones y **bajar** la nota. Pasa con
  «balón en vuelo perfecto» (−0,04 con la GT controlada).

### Contexto de cada posesión
Para entender en qué situaciones fallamos, a cada posesión real se le asigna:
- su duración;
- qué parte cae en un hueco largo de balón;
- cómo empieza: tipo de saque, o tipo de robo según el evento de Metrica;
- cómo acaba: tipo de parón, o tipo de pérdida;
- cuántos pases da el equipo;
- en qué tercio del campo empieza.

---

## 2. Hallazgo 0: el banco medía otra definición de posesión

### Qué dice el reto
`CHALLENGE.md` define el cambio de posesión como el momento en que el rival gana el balón
**controlado**. No cuentan:
- un toque sin control;
- un desvío de un pase que no lo intercepta;
- un duelo aéreo ganado sin control posterior.

Como pistas de control sugiere que el jugador conserve el balón unos segundos o que el equipo dé
más de dos pases. Y añade: si tu lectura de la definición no coincide con la GT pública, **manda
la GT pública**.

### Qué hacía nuestro banco
La GT del banco sale de los eventos de Metrica (ADR-0006). Cada RECOVERY o BALL LOST cambiaba la
posesión, aunque el balón volviera al primer equipo un segundo después. Resultado: muchas
«posesiones» de 1-2 s que el reto no consideraría posesiones.

### La regla corregida
`hackbarca.io.external.controlled_possession(gt, min_frames)`:
- un robo solo cambia la posesión si el nuevo equipo conserva el balón al menos T segundos; si no,
  esos frames siguen siendo del equipo anterior;
- tras un balón muerto, la posesión vale siempre, por corta que sea: al reanudar hay un poseedor
  claro.

```
 GT por eventos:  H H H H H H H H | A A | H H H H H H H H     → 3 posesiones (una de 2 frames)
 GT controlada:   H H H H H H H H   H H   H H H H H H H H     → 1 posesión  (el robo no se controló)
```

### Cómo elegimos T
Buscamos el T con el que el banco tiene la misma forma que la GT pública del reto:

| T | posesiones | media | mediana | < 3 s | spellF1 del modelo actual (sin reentrenar) |
|---|---|---|---|---|---|
| 0 (GT por eventos) | 1.138 | 10,4 s | 4,8 s | 35,9 % | 0,542 |
| 1 s | 956 | 12,4 s | 6,8 s | 24,2 % | 0,600 |
| **1,5 s** | **868** | **13,7 s** | **8,4 s** | **17,5 %** | **0,628** |
| 2 s | 802 | 14,8 s | 9,5 s | 13,3 % | 0,640 |
| 3 s | 685 | 17,3 s | 11,9 s | 5,4 % | 0,661 |
| **GT pública del reto** | 17 en 5 min | **14,2 s** | — | **17,6 %** (3 de 17) | — |

Con **T = 1,5 s** el banco reproduce casi exactamente la media y el porcentaje de posesiones cortas
de la GT pública.

### Por qué no es hacer trampa
- **No usamos el spellF1 de los 5 min públicos para elegir nada**, como manda ADR-0008. Solo la
  forma de su distribución de duraciones, que es justo lo que `CHALLENGE.md` dice que hay que
  mirar para entender la definición.
- La regla sale del texto oficial, no de un ajuste ciego: es la pista «conserva el balón unos
  segundos».
- **Ojo**: la nota sube de forma natural al subir T, porque hay menos posesiones cortas difíciles.
  Por eso T no se elige por la nota, sino por el parecido con la GT pública.

### Qué cambia
- **Unas 270 de las 614 posesiones que dábamos por perdidas (~44 %) eran robos sin control** que
  en el reto no existen.
- **¿Ayuda reentrenar con la definición correcta? No (E021).** Con el banco controlado (T = 1,5 s),
  ajustando Viterbi siempre dejando un partido fuera:

  | modelo | partido 1 | partido 2 | partido 3 | agregado |
  |---|---|---|---|---|
  | apilado actual (GT por eventos), Viterbi reajustado | 0,646 | 0,592 | 0,634 | **0,6245** |
  | apilado reentrenado con la GT controlada | 0,635 | 0,588 | 0,642 | 0,6225 |

  −0,002 y mejor solo en 1 de 3: no pasa ADR-0008. El modelo ya se comportaba casi como pide la
  definición, porque Viterbi cobra por cada cambio y absorbe los robos cortos. Lo que estaba mal
  era **el examen**, que le contaba como fallos robos de 1 s. Se mantiene el modelo de ADR-0011 y
  la GT controlada se usa **para evaluar**.
- **¿Sigue calibrado el banco?** El baseline oficial en el banco con las dos verdades, frente al
  reto real (los 4 números publicados):

  | variante | GT por eventos | GT controlada | reto real |
  |---|---|---|---|
  | raw | 0,052 | 0,040 | 0,049 |
  | s25 | 0,197 | 0,194 | 0,200 |
  | s50 | 0,267 | 0,280 | 0,243 |
  | s125 | 0,242 | 0,275 | 0,224 |

  Se mantiene el orden (s50 > s125 > s25 > raw, aunque s50 y s125 casi empatan) y el banco queda
  unas **4 centésimas más fácil** que el reto (antes, 2-3). No hace falta recalibrar. El macroF1
  del banco v2.1 (0,61) está por debajo del real (0,67): ya pasaba con la GT por eventos y es un
  pendiente menor.
- **Estimación orientativa en el reto**: 0,625 − ~0,04 ≈ **0,55-0,60**, frente al 0,243 del
  baseline. Antes estimábamos ~0,45-0,50. Es muy incierta (±0,05).
- Todo número anterior (E001-E019) está medido con la GT por eventos. **No se compara con los
  nuevos.**
- Incertidumbre: 17 posesiones son pocas, así que la media de 14,2 s tiene mucho error. Por eso se
  comprobará con T = 1 y T = 2, y DFL, cuya GT sale de tracking corregido como la del reto, servirá
  de verificación.

**El resto del documento usa la GT controlada (T = 1,5 s).** La §8 compara con la GT por eventos.

---

## 3. La nota y qué tipo de error domina

**spellF1 0,628**, con **738** posesiones predichas frente a **868** reales.

| umbral IoU | precisión | recall |
|---|---|---|
| 0,3 | 0,80 | 0,68 |
| 0,5 | 0,71 | 0,60 |
| 0,7 | 0,52 | 0,45 |

**Cómo leerlo:**
- **Precisión**: de las posesiones que decimos, cuántas son reales.
- **Recall**: de las reales, cuántas encontramos.
- La precisión es mayor que el recall en todos los umbrales, y predecimos 130 posesiones menos de
  las que hay. **Nuestro defecto es no ver posesiones (fusionar de más), no inventarlas.**
- La caída fuerte de 0,3 a 0,7 indica que muchas de las que encontramos tienen las fronteras algo
  movidas. Pero eso es secundario: ver §5.

---

## 4. Por qué se pierde cada posesión

**Se pierden 344 de las 868 posesiones reales (39,6 %).**

| causa | nº | % de las perdidas | duración mediana | qué significa |
|---|---|---|---|---|
| **equipo contrario** | 145 | 42 % | 3,3 s | En ese tramo dimos el balón al rival: no vimos que esta posesión existía |
| **fusionada: posesión rival corta no detectada** | 79 | 23 % | 6,7 s | El equipo era correcto, pero se juntó con la siguiente porque la del rival en medio desapareció |
| **fusionada: ambos** (rival y parón en medio) | 49 | 14 % | 4,9 s | Igual, con también un parón en medio |
| tapada por D | 33 | 10 % | 3,5 s | Creímos que el balón estaba muerto con el juego vivo |
| fronteras desplazadas o recortadas | 18 | 5 % | 4,1 s | Encontrada, pero con un solape de menos del 50 % |
| partida en trozos | 11 | 3 % | 36 s | Una posesión larga rota en varias |
| fusionada: parón no detectado | 9 | 3 % | 4,3 s | Faltó el D que la separaba de la siguiente |

### El mecanismo principal, dibujado

Un robo que no vemos no cuesta una posesión, **cuesta tres**:

```
 realidad:  H H H H H H H H | A A A A | H H H H H H H H    → 3 posesiones: H1, A, H2
 nosotros:  H H H H H H H H   H H H H   H H H H H H H H    → 1 posesión larga de H

 · A desaparece                → «equipo contrario»
 · nuestra H larga solapa H1 en 8/20 = 0,4 < 0,5, y H2 igual
                               → H1 y H2 «fusionadas: posesión rival no detectada»
 · además nuestra H larga no empareja con nada → 1 sobrante
```

«Equipo contrario» y «fusionada» (42 % + 39 %) son **dos caras del mismo fallo: no detectar un
cambio de posesión**. Juntas suman el **79 % de las pérdidas**.

### Posesiones que nos sobran (falsos positivos)

| tipo | nº | qué son |
|---|---|---|
| solapa una real pero mal ajustada | 122 | Sobre todo las posesiones largas fusionadas del dibujo |
| equipo inexistente | 67 | Le dimos el balón a un equipo que no lo tenía en ese tramo |
| trozo corto (< 3 s) de una posesión real | 25 | Saltos sueltos que Viterbi no eliminó |

---

## 5. En qué situaciones fallamos

### Por duración

| duración | posesiones | encontradas |
|---|---|---|
| < 3 s | 152 | 24 % |
| 3-10 s | 331 | 52 % |
| 10-30 s | 275 | 80 % |
| > 30 s | 110 | 87 % |

Las cortas son difíciles por naturaleza: hay pocos frames de evidencia y Viterbi cobra por cada
cambio. **El grueso de las pérdidas en número está en 3-10 s** (158), no en las ultracortas (116).

### Por huecos de balón

| parte de la posesión en un hueco largo (> 4 s) | posesiones | encontradas |
|---|---|---|
| nada | 547 | 64 % |
| hasta la mitad | 113 | 84 % |
| **más de la mitad** | 208 | **37,5 %** |

La fila del medio sale alta porque las posesiones largas tienen más probabilidad de tocar un hueco
y son más fáciles. **Comparando a igual duración** (GT por eventos), el efecto del hueco es claro:

| duración | sin hueco | más de la mitad en hueco |
|---|---|---|
| 3-10 s | 57 % | 33 % |
| 10-30 s | 80 % | 67 % |
| > 30 s | 96 % | 57 % |

### Por cómo empieza la posesión

| inicio | n | encontradas | lectura |
|---|---|---|---|
| saque de centro | 18 | 89 % | Poseedor obvio |
| falta | 92 | 80 % | Poseedor claro tras un parón |
| saque de banda | 110 | 67 % | Claro, pero suele ser corta |
| robo con balón suelto | 220 | 58 % | Robo en juego: difícil |
| **intercepción** | 305 | **53 %** | El tipo de robo más común y de los peores |
| **córner** | 26 | **42 %** | El área llena de gente: la cercanía engaña |

Las reanudaciones tienen un poseedor claro. **Los robos en juego, no.** Es coherente con §4.

### Por cómo acaba

| final | encontradas |
|---|---|
| tiro | 64-70 % |
| pérdida por robo | 67 % |
| balón fuera | 65 % |
| pase interceptado | 62 % |
| falta | 61 % |
| cabezazo | 44 % |
| **despeje** | **12,5 %** |

Los finales «aéreos» (cabezazos, despejes) son el punto débil. En 2D no vemos la altura del balón,
y un despeje lo cambia de sitio sin que nadie lo controle.

### Por número de pases del equipo

| pases | posesiones | encontradas |
|---|---|---|
| 0 | 229 | 34 % |
| 1-2 | 284 | 54 % |
| 3-5 | 195 | 77 % |
| 6 o más | 160 | 91 % |

Las posesiones de juego elaborado las clavamos. Las de recuperar y perder enseguida, no.

### Por zona del campo
Apenas importa: 58 % en el primer tercio, 66 % en el central y 56 % en el último. El último tercio
es algo peor, por el área llena de gente.

---

## 6. Fronteras: ¿llegamos tarde o no vemos el cambio?

Para cada transición real se busca una predicha **del mismo tipo** cerca.

| transición | n | ≤ 0,5 s | ≤ 1 s | ≤ 2 s | desfase mediano | falsas |
|---|---|---|---|---|---|---|
| **cambio de equipo** (H↔A) | 621 | 27 % | 37 % | **49 %** | +1 frame | 145 |
| parón (H/A → D) | 241 | 48 % | 53 % | 59 % | +1 frame | 145 |
| reanudación (D → H/A) | 241 | 43 % | 56 % | 73 % | 0 | 109 |

- **No llegamos tarde.** Cuando detectamos un cambio, lo colocamos en su sitio: el desfase mediano
  es de 0-1 frame (40 ms). Viterbi, que ve la mitad entera de una vez, hace bien su trabajo.
- **La mitad de los cambios de equipo no tienen ninguna frontera nuestra a menos de 2 s.** No es
  que lleguen tarde: **no los vemos**.
- Los parones se detectan peor que las reanudaciones. Al reanudar, el poseedor se ve claro; al
  morir el balón, en cambio, suele estar perdido en un hueco. Además hay 145 parones falsos,
  sobre todo en huecos largos con el juego vivo.

---

## 7. Frame a frame

### Según haya balón

| contexto | % de frames | acierto | cómo se confunde |
|---|---|---|---|
| balón visible | 61 % | **0,916** | H↔A el 8 %; casi nunca con D |
| hueco corto (≤ 4 s) | 1 % | 0,896 | — |
| **hueco largo (> 4 s)** | **38 %** | **0,778** | Con el juego vivo: **D el 22 %** y el otro equipo el 12-13 %. Con el balón muerto: vivo el 13 % |

En los huecos largos el modelo duda sobre todo entre «sigue el juego» y «balón parado». Cuando
predice D en medio de una posesión, la parte en dos.

### Según la fase del juego (frames con el juego vivo)

| fase | frames | acierto |
|---|---|---|
| balón en vuelo (durante un pase) | 99.036 | 0,91 |
| resto | 188.094 | 0,83 |
| **± 0,5 s alrededor de un duelo** | 9.890 | **0,70** |

Durante un pase el modelo ya sabe de quién es: el contexto lo resuelve. **Los duelos son el
momento más confuso.**

### La señal del «jugador más cercano»
Es la intuición básica: tiene el balón el equipo del jugador más cercano.

| balón | el más cercano es del poseedor | acierto del modelo cuando el más cercano acierta | … cuando falla |
|---|---|---|---|
| degradado (como el ATD) | 70 % | 0,95 | 0,83 |
| limpio (perfecto) | 79 % | 0,91 | 0,78 |

**Incluso con el balón perfecto, la cercanía engaña uno de cada cinco frames**: pases en vuelo,
presión, rebotes. El modelo ya va más allá de esa intuición y acierta el 83 % cuando la cercanía
engaña. Pero mejorar el balón por sí solo no basta: hacen falta señales de **quién controla**, no
solo de quién está cerca.

### Calibración: ¿se fía el modelo de lo justo?

| probabilidad máxima | % de frames | acierto |
|---|---|---|
| ≥ 0,9 | 80 % | 0,92 |
| 0,7-0,9 | 11 % | 0,67 |
| 0,5-0,7 | 8 % | 0,52 |
| < 0,5 | 1 % | 0,41 |

Bien calibrado en los extremos y algo **sobreconfiado en la franja media**: dice 0,8 y acierta
0,67. La temperatura de Viterbi lo compensa en parte.

---

## 8. Cuánto valdría arreglar cada cosa (oráculos)

spellF1 si la predicción fuera perfecta solo en esos frames. Base: 0,628 (GT controlada).

| escenario | spellF1 | ganancia | lectura |
|---|---|---|---|
| equipo perfecto (todos los frames vivos) | 0,942 | +0,314 | El techo: casi todo el margen está en el equipo, no en el D |
| causa «equipo contrario» resuelta | 0,829 | +0,201 | Ver los robos es lo que más vale |
| **huecos largos perfectos** (balón fusionado ideal) | **0,786** | **+0,158** | El techo del balón fusionado |
| causa «fusionada: posesión rival corta» resuelta | 0,752 | +0,124 | La otra cara de los robos |
| posesiones < 3 s perfectas | 0,727 | +0,099 | — |
| **la mitad de los huecos largos perfectos** | **0,709** | **+0,081** | Balón fusionado realista |
| causa «fusionada: ambos» resuelta | 0,709 | +0,081 | — |
| D perfecto (frames D) | 0,669 | +0,041 | El balón muerto importa, pero menos |
| causa «tapada por D» resuelta | 0,656 | +0,028 | — |
| partida en trozos / fronteras desplazadas | 0,64 | +0,01 a +0,015 | Poco que ganar |
| huecos cortos perfectos | 0,627 | 0 | Nada que ganar |

Recuerda: las ganancias **no se suman**, porque varias filas atacan el mismo fallo.

---

## 9. Comparación: GT por eventos frente a GT controlada

El mismo modelo y las mismas predicciones, medidos con las dos verdades:

| | GT por eventos | GT controlada (1,5 s) |
|---|---|---|
| posesiones reales | 1.138 | 868 |
| predichas | 738 | 738 |
| **spellF1** | **0,542** | **0,628** |
| precisión / recall (IoU 0,5) | 0,71 / 0,46 | 0,71 / 0,60 |
| posesiones perdidas | 614 (54 %) | 344 (40 %) |
| equipo contrario | 303 | 145 |
| fusionadas (las tres) | 233 | 137 |
| tapada por D | 42 | 33 |
| < 3 s encontradas | 10,5 % | 23,7 % |
| cambios H↔A a ≤ 2 s | 40 % | 49 % |
| oráculo huecos ideales | +0,165 | +0,158 |
| oráculo mitad de huecos | +0,084 | +0,081 |
| oráculo D perfecto | +0,037 | +0,041 |
| oráculo equipo perfecto | +0,405 | +0,314 |

- **La precisión no cambia (0,71) y el recall sube de 0,46 a 0,60.** Las posesiones que
  «perdíamos» de más eran justo los robos de 1 s que el reto no cuenta.
- Las conclusiones de fondo **no cambian**: el problema son los robos no vistos y los huecos. Solo
  cambia su tamaño.

---

## 10. De las causas a las soluciones

| # | causa raíz | evidencia | solución propuesta | por qué debería funcionar | cómo se valida | ganancia esperada |
|---|---|---|---|---|---|---|
| 1 | **La GT del banco no seguía la definición del reto** | 36 % frente a 18 % de posesiones < 3 s | GT controlada (ADR-0012) **para evaluar** | Mide lo que puntúa el reto | **Hecho.** E021: reentrenar no aporta (0,6225 frente a 0,6245), así que el cambio es solo de medida | Ninguna en el modelo; el examen pasa de 0,542 a 0,625 |
| 2 | **No vemos la mitad de los robos** | 49 % de los H↔A a ≤ 2 s; 79 % de las pérdidas | Señales específicas de cambio de posesión: (a) **reacción colectiva**: al robar, un bloque se abre y adelanta y el otro se cierra y repliega (v4 dio +0,009 con 3 partidos: reintentar con 10 y la GT nueva); (b) **sentido del balón respecto a los equipos** (hacia qué portería avanza tras el robo); (c) **modelo por segmento**: proponer tramos candidatos y decidir con features de todo el tramo si son una posesión rival real | El robo cambia el comportamiento de 22 jugadores, no solo del más cercano; esa señal existe aunque falte el balón | ADR-0008 en el banco | La mayor por potencial (oráculo +0,20 a +0,31) |
| 3 | **Huecos largos** | 38 % de los frames, acierto 0,78; vivo → D el 22 % | **Balón fusionado** (broadcast + vídeo, ADR-0010) | Con balón acertamos 0,92 frente a 0,78 | Puerta de cobertura con datos reales (agregados) y simulación en el banco | +0,05 a +0,10 |
| 4 | **Balón parado mal detectado dentro de los huecos** | parones a ≤ 2 s 59 %; 145 falsos | **Detector de balón parado por la colocación**: jugadores quietos, barrera, formación de córner, portero con el balón en el área pequeña, jugador pegado a la banda | Un balón parado tiene una geometría típica, y se ve sin balón | Banco (los eventos marcan el tipo de saque) | Hasta +0,04 |
| 5 | Duelos, aéreos y despejes | 0,70 cerca de un duelo; despejes 12,5 % | Altura y parábola del balón desde el detector de vídeo (tamaño aparente); contexto posterior al duelo | En 2D no hay altura; el vídeo sí la sugiere | Difícil en el banco (Metrica no tiene altura) | Baja-media |
| 6 | Infrasegmentación | 738 frente a 868; precisión > recall | Reajustar Viterbi con la GT nueva; **HSMM** con duración mínima ~1,5 s, coherente con la regla | Viterbi cobra igual cualquier cambio; HSMM sabe qué duraciones son normales | ADR-0008 | +0,01 a +0,03 |
| 7 | Sobreconfianza media | calibración 0,8 → 0,67 | Temperatura (ya se busca) o calibración isotónica | Viterbi pesa mejor la evidencia | ADR-0008 | Pequeña |

**Lo que no merece esfuerzo ahora:**
- ajustar las fronteras al toque: el desfase ya es de 0-1 frame y el ATD apenas muestra toques;
- los huecos cortos (oráculo 0);
- evitar partir posesiones largas (+0,015).

---

## 11. Límites de este análisis

- **Es el banco, no el reto.** Son 3 partidos de Metrica estropeados para parecerse al ATD. La
  forma de los fallos debería trasladarse, pero los números exactos no.
- **La GT del banco sale de eventos**, no de tracking corregido como la del reto. La regla de
  control la acerca, pero no es idéntica. DFL lo verificará.
- **T = 1,5 s se fija con 17 posesiones públicas**: hay incertidumbre, de ahí la comprobación con
  T = 1 y T = 2.
- La taxonomía usa reglas sencillas (umbrales del 50 % y el 70 %). Un caso frontera puede caer en
  una causa vecina, pero el reparto grueso es robusto.
- Los oráculos son techos teóricos, no promesas.

## 12. Pendiente

- ~~E021: reentrenar con la GT controlada~~ → hecho: no aporta, se mantiene el modelo.
- Medir el modelo vigente con T = 1 y T = 2 (`tune_l3 --from-oof ... --controlled 1` y `2`).
- Repetir con DFL (10 partidos) cuando se descargue.
- Repetir cuando entre cada solución, para ver qué causa baja.

## Glosario rápido

- **Posesión**: racha máxima de H o de A; un D la corta.
- **IoU**: solape entre una posesión real y una predicha (intersección / unión).
- **Precisión / recall**: de las que decimos, cuántas son reales / de las reales, cuántas
  encontramos.
- **Fuera de muestra**: predicción de un partido con un modelo que no lo ha visto.
- **Hueco largo**: más de 4 s seguidos sin balón en el tracking.
- **Oráculo**: sustituir la predicción por la verdad en ciertos frames para medir cuánto valdría
  arreglarlos.
- **GT controlada**: verdad en la que un robo solo cuenta si el nuevo equipo conserva el balón
  ≥ 1,5 s.
