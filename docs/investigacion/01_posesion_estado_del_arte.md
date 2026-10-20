# 01 — Estimar la posesión por frame: estado del arte y opciones

_Autor: Manel (con Claude) · 2026-10-20 · Estado: investigación previa a ver los datos. Hay que revisarla en cuanto se resuelvan Q1-Q8 de `00_reto.md`._

## 1. Entender bien el problema

**Posesión de equipo ≠ jugador más cercano al balón.** Es la trampa principal:
- Durante un **pase en vuelo**, el balón no está cerca de nadie, o está cerca de un rival, pero la posesión sigue siendo del equipo que pasa hasta que alguien del otro equipo lo **controla**.
- Un **duelo o rebote** no es un cambio de posesión si el rival no llega a controlar el balón. Un toque defensivo sin control (despeje, bloqueo) tampoco lo es siempre; depende de cómo lo etiquete la organización.
- **Balón muerto** (fuera, falta, gol, lesión): puede ser una clase propia o no (Q1).

Por tanto, el problema es de **segmentación temporal**: detectar los instantes de control y propagar la posesión entre ellos. No es una clasificación frame a frame independiente. Cualquier modelo que no tenga memoria temporal va a "parpadear" en los pases y perderá precisión justo en los frames que distinguen a los equipos buenos.

## 2. Literatura relevante

| Trabajo | Idea | Resultado | Qué nos llevamos |
|---|---|---|---|
| **Link & Hoernig et al. (2016)**, *Detection of individual ball possession in soccer* | Física del balón (cambios de velocidad/dirección) + ML para detectar control individual | Precisión/recall 80-92 % con datos limpios; unos 4 pp menos con datos sin corregir | Definir el **control** como evento físico (toque), no como proximidad |
| **Vidal-Codina et al. (2022)**, *Automatic event detection in football using tracking data* (MIT/FIFA) | Árbol de decisión: modelo de posesión frame a frame + modelo de eventos usando cambios de posesión y reglas del juego | >90 % en la mayoría de eventos | Implementado en **DataBallPy** (`add_team_possession`): radio de 1,5 m, umbral de velocidad del balón de 5 m/s y de ángulo de 10°. **Baseline listo para usar** |
| **Borghesi et al. (2023)**, *Using TCNs to estimate ball possession in soccer games* (ESWA) | Red convolucional temporal sobre posiciones de jugadores y balón; posesión + vivo/muerto | **86,2 %** de acc. en posesión (+7,3 pp frente a reglas) y 89,2 % en vivo/muerto | Un modelo **secuencial** supera a las reglas. Referencia de lo que es una "buena" accuracy |
| **arXiv 2306.08206** (Set Transformers) | Representación invariante al orden de los jugadores; predice el poseedor como paso intermedio | — | Arquitectura si queremos *deep learning* |
| **arXiv 2501.10299** (embedding con transporte óptimo) | Predecir qué equipo tiene el balón **solo con las posiciones de los jugadores** | 81,8 % con embedding centrado (59,7 % con coordenadas crudas) | **La forma del equipo por sí sola ya da más del 80 %.** Clave cuando el tracking del balón falla |

## 3. Señales disponibles y su valor esperado

1. **Ball–jugador** (distancia al jugador más cercano de cada equipo, quién está dentro de un radio de control): la señal más directa, pero ruidosa cuando el balón está mal trackeado, en el aire o en pases.
2. **Cinemática del balón** (aceleración, cambio de ángulo, velocidad): detecta **toques**. Un toque = posible inicio de posesión de ese equipo.
3. **Forma colectiva** (anchura y profundidad del equipo, altura de la línea defensiva, centroides, velocidad media hacia la portería rival): el equipo con balón se abre y el que defiende se compacta. Es robusta frente a fallos del balón.
4. **Eventos Smart Tagging** (si están en el partido evaluado, Q5): cada pase, recepción o tiro lleva su equipo. Rellenar entre eventos ya da una posesión muy buena; el reto es **sincronizar** bien (offset temporal) y corregir los errores de un etiquetado automático.
5. **Vídeo**: en la fase 1 no hace falta (dan el tracking), pero es útil para **etiquetar a mano** unos minutos y validar.

## 4. Escalera de soluciones (de menos a más)

| Nivel | Método | Coste | Comentario |
|---|---|---|---|
| L0 | Vecino más cercano por frame | 1 h | Solo como suelo de referencia |
| **L1** | **Reglas tipo Vidal-Codina** (radio + detección de toques) + **histéresis** (la posesión se mantiene hasta que el rival controla) + regla de balón muerto (fuera del campo / balón parado) | 1 día | **Primer envío.** Probablemente ya entra en el top 20 si la métrica es accuracy |
| L2 | L1 + **fusión con eventos** Smart Tagging sincronizados | 1-2 días | Si Q5 = sí, es probablemente el salto más grande |
| L3 | **Clasificador por frame** (LightGBM) con features 1-4 en ventanas temporales + **suavizado HMM/Viterbi** con penalización de cambio | 2-3 días | Necesita etiquetas: reales (Q4), *weak labels* de los eventos o etiquetado manual con el vídeo |
| L4 | **Modelo secuencial** (TCN / BiGRU / Transformer) sobre la secuencia de frames | 3-5 días | Estado del arte (Borghesi). Solo merece la pena si hay suficientes etiquetas |
| L5 | *Ensemble* L2+L3/L4 con post-proceso de duración mínima de posesión | — | Rematar en los últimos días |

**Recomendación**: hacer L1 rápido y enviarlo pronto (para conocer la métrica y el formato), y en paralelo L2 y L3. No empezar con L4 sin haber visto los datos y las etiquetas.

## 5. Validación (cómo sabremos si mejoramos sin quemar envíos)

- Si hay etiquetas (Q4): validación por **bloques temporales** (por partes o tramos de 5 min), nunca con *split* aleatorio por frame, porque los frames vecinos son casi idénticos y eso da *leakage*.
- Si no las hay: **etiquetar a mano 10-15 min** de vídeo repartidos entre las dos partes y distintas zonas del campo (cada uno 5 min, con una herramienta sencilla de marcar cambios de posesión). Es caro pero es lo que nos separa de equipos que van a ciegas.
- Analizar los errores por tipo: pases largos, duelos, balón muerto, frames sin balón y transiciones. Las mejoras de verdad se esconden en las **transiciones**.
- **Calibrar el desfase**: un error típico es detectar el cambio de posesión unos frames tarde o pronto de forma sistemática. Hay que medir el error en frames alrededor de los cambios.

## 6. Herramientas

- **kloppy**: carga estándar del formato Metrica (EPTS FIFA / JSON) y de otros proveedores.
- **DataBallPy**: posesión de equipo ya implementada (Vidal-Codina), sincronización tracking-eventos.
- **mplsoccer**: dibujo del campo y animaciones para depurar.
- **pandas/polars + numpy**, **LightGBM**, **PyTorch** (si llegamos a L4).
- Datos de muestra públicos de Metrica (`github.com/metrica-sports/sample-data`) para **montar el pipeline antes de tener los datos oficiales**.

## Fuentes
- Hoernig, Link et al. — https://portal.fis.tum.de/en/publications/detection-of-individual-ball-possession-in-soccer/
- Vidal-Codina et al. — https://arxiv.org/abs/2202.00804
- Borghesi et al. (TCN) — https://iris.polito.it/handle/11583/2976470
- Set Transformers — https://arxiv.org/abs/2306.08206
- Optimal transport embedding — https://arxiv.org/pdf/2501.10299
- DataBallPy — https://databallpy.readthedocs.io/en/latest/api_docs/tracking_data_api.html
- Metrica sample data — https://github.com/metrica-sports/sample-data
