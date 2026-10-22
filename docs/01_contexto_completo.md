# Contexto completo del proyecto, desde cero

> Para quien llega sin saber nada: compañeros del equipo o un agente en una sesión nueva. Resume
> el reto, los datos, cómo validamos, cómo funciona el modelo, dónde fallamos y qué viene. Las
> cifras salen de `docs/experimentos.md` y de los ADR; ante cualquier duda, mandan esos ficheros.
> Actualizado: 2026-10-22.

## 1. El reto

- **More Than A Hack 2027**, fase 1 (Metrica Sports). Más de 10.000 inscritos; **pasan 20**.
- Somos **«La MasIA»** (3 personas). **Entrega: 17 nov 2026.**
- Entrada: un partido, **Barça–Athletic**. Vídeo táctico a 25 fps: **199.319 frames**, de los que
  158.857 se puntúan.
- Salida: una etiqueta por frame.

| etiqueta | significado |
|---|---|
| **H** | posesión del Barça (local) |
| **A** | posesión del Athletic |
| **D** | balón muerto |
| **X** | sin respuesta; fuera de las partes no cuenta |

- Entregables: `La MasIA.csv`, `La MasIA.pdf` (documento técnico) y el repo **público** que
  regenera **el mismo CSV byte a byte** con un comando.

### La métrica: spellF1 (lee esto antes de proponer nada)

- **No puntúa frames, puntúa posesiones.** Una posesión es una racha máxima de H o de A, y un D en
  medio la corta: «H D H» son dos.
- Una posesión predicha acierta si **se solapa (IoU) con una real** por encima de un umbral. Se
  promedia en IoU 0,3-0,7, y el F1 combina precisión y recall.
- Consecuencias:
  - **partir una posesión o fusionar dos se paga muy caro**;
  - **llegar 1 s tarde a un cambio cuesta ~35 %**, porque hay muchas posesiones cortas;
  - acertar frames no basta: el baseline oficial tiene macroF1 0,67 y spellF1 solo **0,243** en
    el partido real.

### Qué cuenta como posesión (CHALLENGE.md)

- Cambia cuando el rival gana el balón **controlado**. Un toque, un desvío o un duelo aéreo sin
  control **no** cuentan.
- La GT sale del tracking corregido de forma mecánica y la referencia es la GT pública de 5 min:
  17 posesiones de 14,2 s de media.
- **Esto importa mucho**: ver §6 y ADR-0012.

### Reglas que nos condicionan

1. **Los agentes de IA no ven los datos del reto** (ADR-0004). El agente escribe el código, las
   personas lo ejecutan en local y solo se comparten agregados (notas, conteos).
2. **No hay leaderboard.** La GT completa es privada; solo tenemos 5 min públicos.
3. **Prohibido etiquetar a mano**: las etiquetas salen de un algoritmo.
4. Prohibido usar otras fuentes de posesión de este partido. Los datos no van al repo.

## 2. Los datos

| fuente | qué es | estado | uso hoy |
|---|---|---|---|
| **ATD táctico** | posición (m) y velocidad de jugadores y balón por frame, sin identidades (46 slots) | ✅ | base de todo |
| **Vídeo táctico** | 1080p con homografía válida en el 100 % de los frames | ✅ | detector de balón listo, sin ejecutar entero |
| **Broadcast** | cámara de tele con su ATD; no sincronizada, con cortes y repeticiones | falta `atd.txt` (parte -009) | alineación programada |
| **Smart Tagging** | eventos automáticos (balones parados…) | ✅ | descartado: +0,008, dentro del ruido (ADR-0007) |
| **GT pública 5 min** | 17 posesiones | ✅ | solo prueba de humo; **nunca elige modelos** |
| **Metrica (externo)** | 3 partidos públicos con tracking y eventos | ✅ | entrenar y validar (el «banco») |
| **DFL (externo)** | 7 partidos de la Bundesliga | ❌ descarga manual | banco de 10 partidos |

**Problemas del ATD:**
- **Falta el balón en el 40,7 %** de los frames puntuados, sobre todo en ~134 huecos de más de 4 s
  (mediana ~11 s). Más de la mitad de esos frames son D.
- **La trayectoria del balón está filtrada**: casi no se ven los toques (0,02-0,04 por segundo).
- Con el balón parado los jugadores corren menos (0,97 frente a 1,62 m/s), y eso ayuda a detectar
  D.

**Cómo se combina todo** (ADR-0010): balón fusionado con prioridad ATD → broadcast alineado →
detector en el vídeo, con una columna que indica la fuente. **Todavía no está conectado al
modelo.**

## 3. Cómo validamos sin leaderboard

- **Banco**: los partidos de Metrica con su GT sacada de eventos, **estropeados para parecerse al
  ATD**: huecos largos de balón, balón suavizado, trayectorias partidas, ruido.
- **Calibración con dos anclas**:
  1. el banco reproduce el orden y la escala de los 4 números publicados del baseline oficial
     (s50 > s125 > s25 > raw);
  2. agregados del ATD real (% de balón ausente, huecos, dinámica del balón, toques).
- **Dejando un partido fuera**: se entrena con 2 y se puntúa el 3.º, en las tres combinaciones.
- **Regla anti-suerte** (ADR-0008): un cambio solo se acepta si sube la nota agregada ≥ 0,01
  **y** mejora en todos los partidos. Todo se registra, también lo que falla.
- **Prueba de robustez**: se evalúa también con otra calibración del balón, igual de compatible
  con el ATD real.
- **Límites**:
  - el banco es ~0,05 más fácil que el reto: sirve para ordenar modelos, no para predecir la nota;
  - solo hay 3 partidos (ruido ±0,02);
  - la GT por eventos no es la del reto (ver §6).

## 4. El modelo actual: L3 apilado (ADR-0011)

1. **Preparar**: mitades por separado; huecos de balón ≤ 0,4 s interpolados; sin identidades.
2. **~65 números por frame** (`features/frame_features.py`, versión 3):
   - cercanía al balón por equipo;
   - estado del balón (visible, velocidad, distancia a la banda);
   - **toques con umbral relativo al partido** (último y siguiente toque y su equipo, receptor);
   - colectivo (velocidad, altura del bloque, dispersión, presión);
   - contexto del hueco (quién estaba cerca justo antes y justo después).

   Se usan ventanas de ±0,5 / 2 / 5 s hacia atrás y hacia delante.
3. **Capa 1: LightGBM** (127 hojas, 400 rondas) → P(H), P(A), P(D). Aumento de datos con los
   equipos intercambiados y predicción simetrizada.
4. **Capa 2 (apilado)**: otro LightGBM que ve las probabilidades de la capa 1 resumidas en ±1, 2,
   5 y 10 s. Se entrena con probabilidades fuera de muestra y elimina los saltos sueltos.
5. **Viterbi**: la secuencia más probable de cada mitad, con coste por cambio (H↔A 20, entrar o
   salir de D 10). Sin retraso, porque decide la mitad entera de una vez.
6. **Formato**: X fuera de las partes, validador oficial, sha256. La entrega no reentrena: el
   modelo va versionado.

**Pendiente**: el artefacto apilado no está entrenado todavía. El CSV de reserva sale de un modelo
en cuarentena (features v2).

## 5. Resultados y qué aportó cada pieza

| modelo | banco v2 | banco v2.1 (GT por eventos) | 5 min reales (humo) |
|---|---|---|---|
| baseline oficial | 0,300 | — | 0,326 |
| L1 reglas + Viterbi | 0,326 | — | 0,469 |
| L3 | 0,502 | 0,465 | 0,588 |
| L3 + toques relativos (v3) | — | 0,513-0,519 | — |
| **L3 apilado** | — | **0,542** (0,558 con el dominio desplazado) | — |

| pieza | ganancia |
|---|---|
| Viterbi | de 0,107 a 0,555 |
| LightGBM frente a reglas | +0,15 a +0,19 |
| Más capacidad | +0,023 |
| Toques | +0,054 |
| Apilado | +0,029 |

**Lo que no funcionó** (también es información):
- Smart Tagging: +0,008.
- Features v2 con umbral fijo de toque: un espejismo de +0,083, porque el balón real va filtrado
  (ADR-0009).
- v4 (empuje de bloque): +0,009 en 2/3 partidos.
- Aleatorización del dominio sobre el apilado: 0,550, pero no mejora los 3 partidos.

**Estimación en el reto**: ~0,45-0,50 con la GT por eventos, frente al 0,243 del baseline. Es muy
incierta (±0,05).

## 6. Dónde y por qué fallamos (análisis del 22 oct)

Detalle completo en `docs/investigacion/02_analisis_fallos.md`.

0. **La GT del banco medía otra definición.** Contaba cada robo de 1 s, y el reto solo cuenta la
   posesión controlada. Con la regla «un robo cuenta si dura ≥ 1,5 s» (ADR-0012), el banco
   reproduce la GT pública (13,7 s de media y 17,5 % < 3 s, frente a 14,2 s y 17,6 %), y **el
   mismo modelo pasa de 0,542 a 0,628**. E021 mide el reentrenado.
1. **No vemos la mitad de los robos.** El 49 % de los cambios H↔A tienen una frontera predicha a
   ≤ 2 s. Un robo perdido cuesta 2-3 posesiones (la suya y las dos vecinas fusionadas): el **79 %
   de las pérdidas**.
2. **Huecos largos** (38 % de los frames): acierto 0,78 frente a 0,92 con balón, y el juego vivo
   sale D el 22 %. Oráculo: +0,158 con el balón ideal, +0,081 con la mitad de los huecos.
3. **Duelos, aéreos y despejes**: 0,70 de acierto cerca de un duelo; despejes encontrados el
   12,5 %.
4. **Infrasegmentación**: 738 posesiones predichas frente a 868 reales; precisión 0,71 y recall
   0,60.
5. **No son el problema**: el retraso de las fronteras (0-1 frame), los huecos cortos y partir
   posesiones largas.

Un dato útil: incluso con el balón perfecto, el jugador más cercano es del equipo poseedor solo el
79 % de las veces.

## 7. El modelo futuro

```
 ATD + broadcast alineado + detector de vídeo ─► BALÓN FUSIONADO (+ fuente)
 audio (silbato, comentarios; experimental)  ─► señales de D y de equipo
                    ▼
 features + señales de CAMBIO DE POSESIÓN + detector de balón parado en huecos
                    ▼
 capa 1 LightGBM (+ modelo de secuencia) ─► capa 2 apilada ─► HSMM (duración mínima)
                    ▼
 La MasIA.csv
 Entrenado con 10 partidos (3 Metrica + 7 DFL) y la GT de posesión controlada.
```

| mejora | ataca | ganancia orientativa |
|---|---|---|
| GT controlada + reentrenar (ADR-0012) | la definición | ya +0,086 sin reentrenar |
| Señales de cambio de posesión | robos no vistos (79 %) | alta, por medir |
| Balón fusionado (ADR-0010) | huecos | +0,05 a +0,10 |
| Detector de balón parado en huecos | D en huecos | hasta +0,04 |
| DFL (10 partidos) | ruido y datos | +0,02 a +0,04 |
| HSMM | infrasegmentación | +0,01 a +0,03 |
| Audio del broadcast (ADR-0013) | D y equipo en huecos | no medible en el banco |

Las ganancias no se suman: varias atacan lo mismo. Cada pieza pasa la regla anti-suerte antes de
entrar.

## 8. Dónde está cada cosa

- **Plan y prioridades vigentes**: `docs/03_plan_ganar.md` (sección «Estado y prioridades»).
- **Decisiones**: `docs/decisiones/` (ADR 0001-0013, todas en *Propuesta* a falta de revisión).
- **Experimentos**: `docs/experimentos.md`. **Diario**: `docs/LOG.md`.
- **Reto y preguntas abiertas**: `docs/00_reto.md` (Q1-Q7). **Datos**: `docs/datos.md`.
- **Código**:
  - `src/hackbarca/` → `io/` (carga, banco, degradación, broadcast), `features/`, `models/`
    (l1, l3), `decode/` (Viterbi), `vision/` (detector de balón), `pipeline/` (tune_l3,
    train_l3_final, make_submission, failure_analysis, atd_stats…).
- **Comandos clave**:
  - Evaluar en el banco: `uv run python -m hackbarca.pipeline.tune_l3 --no-tags --features 3 --stack [--controlled 1.5]`
  - Análisis de fallos: `uv run python -m hackbarca.pipeline.failure_analysis --controlled 1.5`
  - Entrega (lo ejecuta una persona): `uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"`
