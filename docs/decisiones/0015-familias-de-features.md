# 0015 — Features por familias (balón, jugadores, bloques, estructura, reanudación) en el modelo final

- **Estado**: Propuesta (**pendiente de la ablación por familias** antes de entregar)
- **Fecha**: 2026-10-24
- **Autor/a**: Manel (con agente)
- **Revisado por**: ☐ ☐

## Contexto
El análisis de fallos (E020, E028) dice que el 79 % de las posesiones perdidas son robos que no vemos, que en
los huecos sin balón (38 % de los frames) confundimos juego vivo con balón muerto y que damos menos posesiones
de las que hay. El modelo miraba sobre todo la cercanía al balón y los toques, pero en el ATD el balón falta
justo en los choques y los toques casi no se ven. Lo que delata un robo es el cambio de comportamiento de los
22 jugadores. La literatura lo respalda: en PathCRF lo social es lo que más aporta; la compacidad y la anchura
cambian con y sin balón (AMISCO, LJMU); también hay trabajos de pressing intensity, contrapresión y
formaciones (ver `docs/04_plan_investigacion.md`).

## Opciones consideradas
1. **Seguir con features v3**: 0,6470 en el banco v3.
2. **Versiones numeradas sueltas (v5, v6, v7...)**: difíciles de entender y de medir por partes.
3. **Familias con nombre, pensadas como las mira un entrenador**, medibles por separado. Elegida.
4. **Modelo profundo** (atención entre jugadores + BiGRU, E034) en lugar de, o además de, features
   colectivas.

## Decisión
- `frame_features(version=...)` acepta familias unidas por `+`. La versión 3 (modelo vigente) no cambia.
  Las familias:
  - **balón**: sentido de progresión antes y después, vuelo largo, balón quieto;
  - **jugadores**: quién aprieta, sprints, desmarques, portero (solo si está a < 25 m de su portería);
  - **bloques**: longitud, anchura, superficie, stretch index, altura de líneas, distancia entre líneas,
    adelanto de un bloque sobre el otro, con los niveles relativos al partido;
  - **estructura**: jugadores por detrás del balón y en campo rival, forma propia abierta o cerrada sin
    etiquetas, cambio de forma;
  - **reanudación**: colocación de balón parado.
- Cada señal lleva su **cambio entre el segundo anterior y el posterior** (1 s y 3 s).
- Candidato final: **LightGBM apilado con `touch+ball+players+blocks+structure+restart`** y el Viterbi
  fijo de siempre.

## Justificación
- **E035**: **0,7185** frente a 0,6470 en el banco v3 (+0,072). Mejora en 9 de 10 partidos (solo baja
  J03WR9: 0,661 → 0,631). Las posesiones predichas pasan de 2.298 a 2.669 (GT 2.885).
- **Comprobación de dominio** (`feature_shift`): los cambios antes/después cuadran entre el ATD real y el
  banco (desajuste < 0,05). Los niveles de bloque no cuadraban (0,3-0,4): en el ATD real los equipos salen
  ~5 m más cortos. Por eso se expresan relativos al partido.
- **Alternativas medidas y rechazadas**:
  - coste de cambio dinámico: sin ganancia (E036);
  - modelo profundo sobre las familias: 0,7172 y 0,7134, frente a 0,7185 (E037);
  - comentario en inglés: los parones que comenta ya los detectamos (E033).

## Consecuencias y cautelas antes de entregar
- **Ablación por familias**: quitar una cada vez para saber de dónde sale la mejora. La primera, sin
  «jugadores», está en marcha.
  - Las features del portero están entre las más importantes y en el ATD real su distribución difiere algo
    (desajuste 0,22).
  - Si sin «jugadores» la nota apenas baja, se quita esa familia.
- Descartar que el modelo aproveche artefactos del simulador. Por ejemplo, el balón que se interpola en los
  parones revelados.
- Analizar por qué empeora J03WR9 y repetir el análisis de fallos con el modelo nuevo.
- Prueba de humo en el reto real; `make_submission` dos veces con el mismo sha.
- Si la ablación lo confirma, pasa a *Aceptada* y se regenera `La MasIA.csv`.
