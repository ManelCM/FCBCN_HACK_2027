# Plan de parones: plano de la tele, silbato y huecos por aglomeración (27 oct → congelación del 14 nov)

> Complementa `docs/05_plan_mejora.md`, no lo sustituye. Lo escribe el agente de Manel el 26 oct.
>
> Hay tres líneas que atacan lo mismo: **saber cuándo el balón está muerto** donde el tracking no lo dice, es
> decir, en los huecos del ATD y en los D cortos. Hoy los D cortos cuestan −0,088 en el banco a la tasa del
> reto (E040). Las líneas son:
> - **L1**: el simulador abre los huecos de juego vivo donde el balón está rodeado de jugadores (banco v3.2);
> - **L2**: un detector de silbato rehecho desde cero, porque E029 y E045 no encontraron señal;
> - **L3**: el plano de la realización, versión 2. Parte de U1 (E055-E057) y añade las repeticiones y una
>   calibración sin ninguna GT del reto.
>
> Antes de las tres va **L0**: dejar el proyecto dentro de las reglas actualizadas del repo oficial.

## L0. Cumplimiento de las reglas del 22-23 oct (prioridad 0, 27 oct, ½ día)

El repo oficial tiene dos commits nuevos, `c073f29` (22 oct) y `f8f2629` (23 oct). Nuestro submódulo sigue en
`3de6b60`. Los cambios que nos afectan:
- La ventana pública de 5 min **no es conjunto de ajuste**, y nunca sirve para elegir entre versiones.
- **Cero etiquetas humanas** sobre BAR-ATH, para cualquier fin. Las pseudoetiquetas que genera un algoritmo
  sí están permitidas.
- Etiquetar a mano otros partidos está permitido, en cualquier cantidad, si se documenta.
- El documento técnico tiene que listar todos los conjuntos etiquetados que usemos: fuente, quién etiquetó y
  cuánto.

**Pasos**:
1. Actualizar el submódulo a `f8f2629` y releer RULES, FAQ, DATA, EVALUATION y SUBMISSION.
2. Escribir **ADR-0019, «uso de la ventana pública»**:
   - la ventana pública solo sirve como prueba de humo (formato, que no se rompa nada) y nunca como puerta,
     calibración ni elección;
   - las puertas sobre el reto se miden **solo con pseudoetiquetas** (la predicción segura, pmax ≥ 0,9) y
     agregados del ATD.
3. **Revisar lo hecho con la ventana pública y dejarlo por escrito** en el ADR:
   - **ADR-0012**: T = 1,5 s se fijó con la forma de la GT pública. Hay que justificarlo también con
     CHALLENGE.md (posesión = control) y con el banco, o volver a fijarlo con un criterio que no la use.
   - **E055-E056**: las cifras de AUC de la ventana pública dejan de contar como criterio. U1 pasa igual con
     la predicción segura (0,80, y 0,78 en los huecos), y la simulación ya está calibrada con la predicción
     segura.
   - **ADR-0017 y la tabla de estado** (M1.1): la cifra de 1,86 tramos lentos por minuto de la GT pública se
     queda como contexto y no fija ningún parámetro.
4. **Arranque de la 2.ª parte** (FAQ oficial): los frames 110.808-113.741 puntúan y son todos `D`.
   - Medir con un agregado (`scripts/run_aggregate.sh`) el % de D que da `La MasIA.csv` en ese tramo.
   - Si no es ~100 %, fijarlo desde la documentación oficial en un ADR. No sale de mirar el partido, sino del
     FAQ.

**Puerta**: ninguna, es obligatorio. El riesgo de no hacerlo es la descalificación.

## L1. Banco v3.2: huecos de juego vivo donde el balón está rodeado (27-29 oct)

> **26 oct, E080: ❌ la hipótesis no se cumple; v3.2 en espera.** La velocidad antes de un hueco solo
> calibra el balón, sin tocar la caché. Objetivo: 2,0 m/s.
> - **Aglomeración**: 0,68 → 0,86 m/s.
> - **Velocidad del balón**: 0,85 m/s.
> - **Adelantar el hueco de parón 1,5-3 s**: 1,1-1,3 m/s, con una pérdida que baja de 21,1 a 18,1.
>
> Siguiente: un agregado del ATD con la velocidad en el último segundo antes de los huecos. El adelanto solo
> entra si ese agregado lo apoya y E049 muestra errores en los bordes de los huecos.

**Problema** (abierto en ADR-0017): antes de un hueco, el balón va a 2,0 m/s en el reto y a 0,7 en el banco
v3.1. La configuración v3 abre casi todos los huecos donde el limpio no tenía balón (`dead_weight` 100) y no
da peso a la oclusión (`occlusion_boost` 1). Por eso los huecos de juego vivo del banco caen en sitios al azar
y no donde el tracker pierde el balón de verdad, que es entre piernas, en disputas y en balones divididos.

**Hipótesis**: si los huecos de juego vivo se abren donde hay gente alrededor del balón, el banco aprende
justo la situación que falla en el reto. Esa situación es un robo o una disputa dentro de un hueco, que el
modelo hoy convierte en un D corto.

**Diseño** (`io/degrade.py`, rama `_long_ball`):
1. **Peso de aglomeración** en juego vivo: `w = crowd^k + c`, con `crowd` = jugadores de los dos equipos a
   menos de `r` m del balón limpio. Valores iniciales: `r` = 3 m, `k` = 2, `c` = 0,2. Sustituye a la oclusión
   binaria (`occlusion_radius` / `occlusion_boost`).
2. **Bordes de los huecos de balón muerto con desfase**: el tracker no pierde el balón justo en el silbato.
   El inicio se mueve U(−0,5, +2,5) s y el final U(−0,5, +1,5) s.
3. **Huecos mínimos**: comprobar con el agregado del ATD si existen huecos de menos de ~3 s, porque el
   proveedor podría rellenarlos. Si no existen, se quitan del simulador.
4. Parámetros nuevos en `DegradeParams`, con valores por defecto que dejan v3.1 idéntico (reproducible).

**Calibración** (`pipeline/calibrate_ball.py`, solo agregados del ATD y del banco limpio):
- Objetivos de E013 y E047 (no deben empeorar) más tres nuevos:
  - velocidad del balón antes de un hueco: 2,0 m/s;
  - jugadores a < 3 m de la última posición del balón antes de un hueco (agregado nuevo del ATD, `atd_stats`);
  - fracción de huecos que empiezan con la predicción segura en vivo (pseudoetiqueta, sin GT).
- Rejilla sobre `k`, `c`, `r` y `dead_weight`.

**Experimentos**:
- **E080**: calibración y ADR-0020, «banco v3.2». Las cifras de v3.2 no son comparables con las de v3.1.
- **E081**: el modelo de E049 evaluado en v3.2 (`--eval-degrade v3_2`, como E048). Tiene que bajar en los
  huecos; si no baja, la hipótesis está mal.
- **E082**: reentrenar en v3.2. Se acepta con ADR-0008 frente a E081 y con el guardarraíl de ADR-0018.

**Coste**: 1 día de código y calibración, unas 3 h de caché y evaluación. Va después de E049, que está en
marcha, para no mezclar bancos ni saturar la memoria.

## L2. Silbato, versión 2 (27-28 oct para la puerta; 30 oct-2 nov si pasa)

> **26 oct, E083: ❌ no pasa; L2 cerrada.**
> - No hay líneas fijas en la banda, así que la hipótesis sobre el fallo de E029 no se cumple.
> - Con el umbral pre-registrado salen 44 silbatos: precisión 0,23 (×2,4 sobre el azar) y cobertura de los
>   parones seguros de 0,105.
> - Hay señal, pero casi nunca se oye: a cualquier umbral, más del 70 % de los parones no tienen silbato
>   detectable.
> - Lo de abajo queda como registro del diseño.

**Por qué falló E029**:
- una columna de STFT de 40 ms (resolución de ~25 Hz) en la banda de 2-4,5 kHz;
- tonalidad «pico / mediana» de **un solo tono**;
- sin ninguna condición de inicio.

Hipótesis a comprobar con un agregado: cualquier línea estrecha y **estacionaria** de la mezcla (un pitido de
la emisión, un armónico del equipo de sonido) da tonalidad alta todo el partido y tapa el silbato. E045 usó
la clase genérica *Whistle* de AudioSet, que está pensada para otro tipo de silbidos.

**Qué sabemos de un silbato de árbitro**: el de bola da **2-3 tonos planos y paralelos** entre ~3,3 y 5,5 kHz.
No son armónicos entre sí, y cada pitido dura de 0,2 a 1,5 s.
- El comentario está por debajo de ~3,5 kHz y sus armónicos se mueven con la voz.
- La grada es de banda ancha.

Nada más en la mezcla tiene esa forma.

**Diseño** (`audio/whistle_v2.py`; no toca `whistle.py`, para que E029 siga siendo reproducible):
1. Audio mono a 16 kHz; STFT con `nperseg` 2048 (bins de ~8 Hz) y salto de 256 muestras (16 ms), por trozos de
   120 s con 6 s de solape.
2. Por cada bin de 3,3-5,5 kHz:
   - **prominencia**: dB por encima de la mediana de ±160 Hz, es decir, estrecho en frecuencia;
   - **novedad**: dB por encima de la mediana del mismo bin en ±2 s, es decir, tiene un inicio. **Esto mata
     las líneas estacionarias.**
3. **Línea** = min(prominencia, novedad), con una mediana de ~0,24 s en el tiempo (sostenido a un tono).
4. **Puntuación** por columna: la segunda línea más fuerte que esté a ≥ 250 Hz de la primera. Exige **dos
   tonos a la vez**.
5. **Evento**: racha con puntuación ≥ umbral (inicial 12 dB) de ≥ 0,15 s; las rachas a < 0,3 s se unen.
6. **Papel del silbato**, sacado del tracking sin etiquetas:
   - **parada**: la velocidad media de los jugadores baja (de 2-4 m/s en los 3 s anteriores a ~1 m/s en los
     2-6 s posteriores);
   - **reanudación**: la velocidad sube;
   - **dudoso**: cambia menos de 0,3 m/s.
7. Paso a frames del ATD con los desfases por parte de E026.

**Puerta (E083, solo agregados y pseudoetiquetas; ninguna GT del reto)**:
- Diagnóstico previo: espectro medio de la banda en todo el partido. ¿Hay líneas estacionarias?
- Eventos por minuto: entre ~1 y ~4. Un partido tiene del orden de 100-150 parones.
- **Precisión**: % de silbatos a ≤ 1 s de un inicio de D seguro o de una reanudación segura, frente al mismo %
  con instantes al azar en vivo seguro. Pasa si es ≥ 3 veces el azar y la precisión es ≥ 0,5.
- **Cobertura**: % de los D largos seguros (≥ 5 s) con un silbato a ≤ 2 s del inicio o del final. Pasa si
  es ≥ 0,4.
- Coherencia con el papel: ≥ 70 % de las «paradas» caen en un inicio de D seguro.
- Se informa la misma tabla para los D cortos: ¿cuántos llevan silbato? Un silbato apunta a un parón real y
  no a un hueco mal leído.

**Si pasa: entrada al modelo como familia `whistle`**, igual que se hizo con `broadcast` (E056-E057):
- **Simulación en el banco** (`io/whistle_signal.py`), porque el banco no tiene audio. Sobre la GT del banco:
  - silbato de parada en el inicio de un D con probabilidad `p_stop`, con desfase de N(0,4 s, 0,3 s);
  - silbato de reanudación con probabilidad `p_restart`, más alta en los saques de centro y las faltas;
  - falsos positivos en vivo a una tasa de `fp_rate` por minuto.

  Los tres parámetros salen de la puerta del reto: cobertura, precisión y tasa en vivo seguro. Son
  pseudoetiquetas, como en U1.
- **Features**: tiempo desde el último silbato, papel del último silbato, silbatos en ±2 s y el cambio
  antes/después.
- **E084**: evaluación con el protocolo (ADR-0008) más el guardarraíl, y desajuste reto/banco de las columnas
  `wh_*` < 0,25.
- **Validación externa opcional**: etiquetar a mano los silbatos de 1-2 partidos de **otras competiciones**
  con audio (permitido si se documenta) para medir precisión y cobertura reales del detector. Va en la lista
  de conjuntos etiquetados del documento técnico.

**Dependencia**: Q6, si podemos usar el audio del broadcast en la entrega. Mientras no haya respuesta, la
entrega principal no depende del silbato y se prepara una variante sin él.

## L3. Plano de la realización, versión 2 (28 oct-3 nov)

**De dónde partimos**: U1 (E055-E056) usa como señal de balón muerto los jugadores visibles en el broadcast,
es decir, plano abierto frente a corte.
- AUC 0,80 con la predicción segura y 0,78 dentro de los huecos del ATD.
- El simulador `io/broadcast_signal.py` está calibrado y E057 (familia `broadcast`) está en cola.

**Qué añade la versión 2**:
1. **Tipos de plano, no solo «abierto/corte»**:
   - plano abierto;
   - plano medio o primer plano de un jugador;
   - **repetición**, que se detecta por la cortinilla o el logo de entrada y salida, un cambio brusco del
     histograma de color y la velocidad de reproducción;
   - público o banquillo.

   La repetición es casi segura de balón muerto, o de un momento justo antes de él. Es U2 de la tabla de
   estado.
2. **Validez de la homografía del broadcast** por frame, como señal barata de «plano utilizable» (solo
   agregados).
3. **Calibración sin GT del reto**:
   - la forma del proceso (planos abiertos, cortes, repeticiones según el estado) sale de la predicción
     segura (pseudoetiquetas);
   - la ventana pública no se usa, según L0.
4. **Calibración externa opcional**: con las etiquetas públicas de tipos de plano y de eventos de SoccerNet
   (otros partidos), estimar P(tipo de plano | vivo/muerto) y la duración de las repeticiones tras un parón.
   Antes hay que confirmar la licencia y que encaja en las bases (Q3). Sirve para contrastar la simulación con
   una fuente que no es nuestro partido.

**Experimentos**:
- **E085**: puerta de los tipos de plano en el reto, con la predicción segura. ¿Suben la AUC frente a U1 dentro
  de los huecos y en los D cortos?
- **E086**: simulador v2 y familia `broadcast` v2, evaluada con el protocolo frente a E057.

**Dependencia**: la misma Q6, porque usa el vídeo del broadcast en la entrega.

## Orden y calendario

| fechas | qué | entregable |
|---|---|---|
| 27 oct | **L0** completo; puerta del silbato (**E083**), que es barata y solo usa la CPU | submódulo en `f8f2629`, ADR-0019, agregado de la 2.ª parte, informe de E083 |
| 27-29 oct | **L1**: código, calibración (E080), E081 y E082, después de E049 | ADR-0020, banco v3.2 |
| 28-30 oct | **L3**: tipos de plano y repeticiones (E085), en la pasada de vídeo con GPU | informe de E085 |
| 30 oct-2 nov | L2, si E083 pasa: simulación y familia `whistle` (E084) | familia `whistle` |
| 2-3 nov | L3: simulador v2 y familia `broadcast` v2 (E086) | familia `broadcast` v2 |
| 4-7 nov | combinación con lo que haya pasado del plan general; ablación | artefacto candidato |
| 8-14 nov | lo que marca T6 en `05_plan_mejora.md` | entrega |

**Reglas comunes a las tres líneas**:
- toda puerta sobre el reto usa solo agregados y la predicción segura, nunca la GT de la ventana pública;
- cada familia nueva tiene su simulación en el banco y se acepta con ADR-0008 más el guardarraíl de ADR-0018;
- cada señal que dependa del broadcast tiene una variante de entrega sin ella, hasta que Q6 se resuelva;
- todo se registra en `docs/experimentos.md` y en el LOG.

## Cuánto podemos esperar (orientativo)

| línea | banco | reto |
|---|---|---|
| L1 | baja al cambiar de banco y sube al reentrenar | **lo que más puede dar**: ataca directamente los D cortos dentro de los huecos (techo del defecto −0,088) |
| L2 | +0,005 a +0,02 si pasa la puerta | parecido, si la simulación es fiel |
| L3 | +0,01 a +0,03 sobre U1 | parecido; la señal real es más rica que la simulada |
