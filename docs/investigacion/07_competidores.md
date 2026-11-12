# Qué están haciendo los demás equipos (repos públicos, 10 nov 2026)

> Análisis del agente de Manel, 10 nov. Fuente: solo repositorios públicos de GitHub.
> - **Forks del repo oficial**: hay 262, de los que 95 tienen commits propios.
> - **Repos independientes**: 128 encontrados con la búsqueda de GitHub y 115 clonados.
>
> Se leyó solo texto (README, documentos técnicos, registros de experimentos). No se ejecutó nada de esos
> repos ni se abrió ningún CSV ni fichero de datos.
>
> **Ojo: las cifras de cada equipo no son comparables con las nuestras ni entre sí.** Cada uno usa su
> propio banco (partidos, degradación, GT y regla de posesión). Las de la ventana pública de 5 min son
> humo (17 posesiones, ±0,33-0,38).

## 0. Lo más urgente: aclaración de la organización del 3-4 nov (nos afecta)

La organización mandó un **email a todos los participantes** el 3 nov. Varios equipos lo citan el 3, 4 o
5 nov y tres lo copian casi literal:
- `SAM-Analytics-2027/.../rules/CLARIFICATIONS.md`;
- `nilaymastaadmi/.../work/rules-clarification-2026-11-04.md`;
- `verneylmavt/.../ORGANIZER-CLARIFICATION.md`.

No está en el repo oficial, que no tiene issues y cuyo último commit es del 23 oct (`f8f2629`). Dice que
**manda sobre la documentación, los emails y las respuestas anteriores**.

**No se permite:**
- comparar nuestra salida con el vídeo;
- etiquetas a mano u otros datos creados manualmente sobre el partido del reto;
- subir los datos del reto a servicios de IA externos;
- **entrenar, ajustar o calibrar el algoritmo con la GT de posesión o con el vídeo del reto**.

**Se permite:**
- comprobaciones automáticas solo con los datos del reto;
- mirar los datos y el vídeo para entender qué es posesión;
- **usar como entrada el vídeo, el tracking, el Smart Tagging u otros datos generados**, siempre que el
  algoritmo no se haya entrenado con la GT ni con el vídeo;
- entregar ficheros de pasos intermedios;
- cualquier dataset con derechos de uso.

**Entrenamiento y ajuste:**
- si hay que entrenar o reentrenar, solo con datos externos;
- si el algoritmo tiene parámetros que ajustar, se ajustan con datos externos, **nunca** con la GT ni con
  el vídeo.

Varios equipos lo leen así:
- las pseudoetiquetas sacadas del partido del reto **no lo convierten en dato externo**;
- tampoco vale el autoentrenamiento sobre el reto;
- los cálculos fijos en inferencia (normalizar por partido, decodificar) no cuentan como ajuste.

### Impacto en nuestro proyecto
| pieza | qué hace | situación con la aclaración |
|---|---|---|
| **U6, conductor por vídeo (E074-E075), en la entrega actual** | `vision/video_pass.py` usa `data/derived/ball_yolo/.../best.pt`, un **YOLO afinado con recortes del vídeo táctico del reto** y pseudoetiquetas del ATD (E061) | ❌ **Es entrenar con el vídeo del reto.** La entrega E075 + regla de 3 s (sha `d976e575…`) no cumple. |
| simulación de U6 en el banco (`io/video_conductor.py`) | cobertura y acierto medidos en el reto | ⚠️ calibrado con datos del reto (no con la GT); cae con U6 de todos modos |
| equipo por color del torso (U5) | color medio en Lab; equipos asignados sin etiquetas | ⚠️ zona gris: agrupar colores en inferencia no es entrenar, pero conviene documentarlo |
| E070 (cinco familias, sin vídeo) y E049 (banco v3.1, sin vídeo) | solo tracking como entrada; entrenados con Metrica y DFL | ✅ |
| simulador del banco calibrado con agregados del ATD (E006, E013, E047) | da forma a los datos externos de entrenamiento | ⚠️ zona gris aceptable: no es la GT ni el vídeo, y otros equipos hacen lo mismo y lo declaran. **Hay que explicarlo en el documento técnico** |
| E006 (`dead_weight` calibrado en parte con «56 % de D en huecos» de la GT pública) | ya señalado en ADR-0019 | ❌ con la aclaración es calibrar con la GT: sustituir ese objetivo o justificarlo sin la GT antes de entregar |
| regla de D cortos (ADR-0021, k fijado en el banco) | | ✅ k sale del banco; el guardarraíl sobre el reto es una comprobación automática |
| U10, T-DEED preentrenado en SoccerNet sobre el broadcast | modelo externo, sin reentrenar, con el broadcast como entrada | ✅ La aclaración **responde a Q6**: el vídeo, incluido el broadcast, se puede usar como entrada |
| ficheros derivados versionados (Q4) | | ✅ «Providing files of intermediate steps» está permitido |
| agente que ve agregados (ADR-0004) | | ⚠️ prohíbe subir los datos del reto a servicios de IA. Los agregados que ve el agente siguen sin aclarar (Q2) |

**Consecuencia directa:** antes de congelar hay que volver a una entrega sin el YOLO afinado con el vídeo del
reto. Opciones, sin decidir todavía:
- E049 (artefacto v3.1, en `external_data/artefactos/v3_1/`) + regla de 3 s;
- o E070.

Si se quiere conservar U6, hace falta un detector de balón entrenado **solo con vídeo externo** (por ejemplo,
datasets de balón de SoccerNet), evaluado de nuevo.

## 1. Panorama: los equipos más trabajados

| equipo (repo) | enfoque | datos de entrenamiento | validación | cifra que declaran (no comparable) |
|---|---|---|---|---|
| **jekkiechan** (fork, 91 ramas, 131 commits) | GRU conjunto + HGB por fases; decodificador por spells | SkillCorner, **SoccerNet-GAR** (tracking en Hugging Face), PCBAS | limpio + «feed-like» (degradado), puertas fijadas antes | «GAR TEST» 0,70; ventana pública 0,630 |
| **katenaball** (jusstinn, 149 commits) | **dos cabezas: P(vivo) y P(local \| vivo)**; LightGBM (213 col.) + **TCN**, mezcla en logit 0,5/0,5; 5 Hz; Viterbi con transiciones contadas | 30 partidos: DFL, Metrica y 20 SkillCorner (**SkillCorner solo en la cabeza de equipo**) | 28 pliegues dejando un partido fuera; puerta ≥ 70 % de pliegues + IC del bootstrap > 0 | F 0,602 frente a 0,586 (+0,016, 22 de 28) |
| **agamemnon-labs** | CNN 1D dilatada (9 bloques, ~20 s de contexto) + Viterbi + **ajuste de cada cambio al primer toque del equipo nuevo (≤ 0,22 s)**; antes del saque de cada parte, D | 7 DFL + 20 SkillCorner (D = fuera de fase de juego) | 5 pliegues + selección anidada | 0,681 (DFL 0,748, SkillCorner 0,657) |
| **FCB-Hacks** (AyanBhardwaj1) | LightGBM P(muerto) y P(local \| vivo), **pesos iguales por posesión**, Viterbi con **coste de cambio menor que parada + reanudación** | 3 Metrica (GT de eventos) | 3 partidos × 3 semillas de degradación nuevas | 0,768 (banco fácil) |
| **AndaAnda** (JulienErbland) | LightGBM + TCN en conjunto (1/4 cada uno); detectores de toque y de reanudación; antes del saque, D | DFL **re-etiquetado con una regla de control** y 2 copias degradadas (balón muerto que se mueve y que no) | DFL (7 × 3 ruidos) + **PFF Mundial 2022 (64 partidos, solo evaluación)** | 0,664 |
| **Mastaadmi** (nilaymastaadmi) | HistGB (72 features) + Viterbi; v8 rellena el balón con el broadcast alineado | 27: 7 DFL + 20 SkillCorner (**SkillCorner solo con el balón visible**) | 5 pliegues por partido; 4 convenciones de etiqueta | 0,51-0,63 según la convención |
| **Derekko-web** | modelo de «consenso» + «rescate anclado» | — | pliegues por minuto con buffer | 0,68 → 0,73 (protocolo propio) |
| **YonoBengkel/barcelona** | — | 7 DFL | dejando un partido fuera | 0,574-0,631 por partido |
| **ZainnB** | etiquetan a mano **otros** partidos | — | techo humano medido: **0,894** spellF1 | — |
| **Ratnesh-101** | «control sostenido» por contacto con decaimiento exponencial que se congela en los huecos | sintético | — | 0,648 sintético |

Otros con trabajo serio:
- Barca-Hackathon: modelo de control logístico + CNN temporal entrenada en Kaggle.
- animepuika: recuperación del balón por vídeo, +0,02 en humo.
- TrymBardsen: HMM + voto.
- hannannoor: regla de retención.
- ra189zor: eventos de cambio aprendidos.
- Moiz005: posesión visual.

Muchos repos son el upstream con pocos cambios.

**Señales de alarma en otros equipos (no nos afectan, pero dan contexto):**
- algunos se ordenan con la ventana pública como si fuera un ranking («#1 absoluto»);
- KevinNaserwan declara 0,9714 en la ventana pública: casi seguro ajustado a esa GT;
- realtonypark hace «hill-climb a 0,70» sobre ella;
- ZesterJ elige los partidos de entrenamiento por estilo, mirando la ventana.

Con la aclaración, todo eso queda fuera de las reglas.

**Dónde estamos nosotros:**
- en banco, E049 da 0,725 y E075 0,769, en la banda alta de lo que declaran los equipos serios (0,60-0,77), aunque los bancos no se pueden comparar;
- en la ventana pública, E070 dio 0,651, entre los valores más altos que se ven (0,55-0,66), que también es humo.

## 2. Ideas que podemos probar, en orden de valor/coste

### I1. Coste de cambio menor que parada + reanudación (barato: solo redecodificar)
- **Qué**: FCB-Hacks fija cambio 64, parada 40 y reanudación 40, y exige que un traspaso cueste al menos 16
  menos que parar y reanudar. Sin esa regla, «el decodificador puede traspasar la posesión con parones
  falsos».
- **Por qué nos toca**: nuestra decodificación es cambio 20, a D 10 y desde D 10. **H→D→A cuesta lo mismo que
  H→A**, así que el Viterbi no penaliza el atajo por un D corto. Es justo el defecto de los D cortos del reto
  (E041-E043), que hoy tapamos con la regla de 3 s.
- **Cómo**: redecodificar los OOF de E049 con la rejilla restringida a `to_dead + from_dead ≥ switch + m`,
  elegir dejando un partido fuera (ADR-0008), y además el guardarraíl del reto. Son minutos.

### I2. Pesos iguales por posesión al entrenar (barato: un reentrenamiento)
- **Qué**: cada frame pesa 8 s / duración de su posesión, recortado a [0,25, 4]. spellF1 cuenta cada posesión
  una vez, y así las cortas pesan lo que en la métrica. FCB-Hacks mide **+0,016 ± 0,003 (9 de 9)**.
- **Por qué nos toca**: el margen está en los robos cortos que no vemos (+0,146, E040).

### I3. SkillCorner otra vez, pero solo para «quién» (medio)
- **Por qué falló E054**: SkillCorner **no tiene etiqueta D**. Su `possession.group` nulo significa
  «desconocido», no balón muerto (lo documentan jekkiechan y katenaball). Nosotros le dedujimos el D.
- **Qué hacen los que ganan con SkillCorner**:
  - katenaball: dos cabezas, y SkillCorner solo entrena la de equipo;
  - Mastaadmi: SkillCorner solo con el balón visible;
  - agamemnon: D = fuera de una fase de juego.
- **Cómo, en nuestro LightGBM multiclase**: añadir filas de SkillCorner solo donde hay H/A y balón visible,
  sin aportar D; o separar en dos cabezas como katenaball.

### I4. Separar en dos cabezas: P(vivo) y P(local | vivo) (medio)
- Lo hacen katenaball, FCB-Hacks y mantas1stan. Permite usar datos sin D (I3) y desacopla el defecto de los D
  cortos del de equipo.
- Encaja con el apilado: dos LightGBM en la capa 1 y la capa 2 igual.

### I5. Antes del saque de cada parte, D, con una regla algorítmica general (barato; pendiente de L0)
- agamemnon y AndaAnda lo implementan sin frames fijos: se detecta la formación de saque de centro o el primer
  movimiento del balón.
- Es justo lo que ADR-0019 pedía (2.ª parte, 2.934 frames D antes del saque).

### I6. Ajuste de la frontera al primer toque del equipo nuevo (barato)
- agamemnon mueve cada traspaso al primer toque del equipo que entra, en ≤ 0,22 s.
- Nuestro oráculo dice que las fronteras dan como mucho +0,015 (E040). Se prueba sobre los OOF.

### I7. Un balón muerto que se mueve en el simulador (medio)
- AndaAnda: en DFL el balón muerto está aparcado, pero en el ATD real se lo llevan a sacar.
- Entrenan con 2 copias por partido, una con el balón muerto moviéndose y otra con él parado. Ataca la
  confusión vivo/muerto con el balón lento (E043), que es el origen de los D cortos.
- katenaball hace algo parecido con sus «mundos» de degradación, que acoplan la pérdida del balón a los
  parones, meten cambios de identidad y suavizan.

### I8. TCN + LightGBM mezclados en logit (caro)
- katenaball +0,016 (22 de 28) y AndaAnda lo usan; agamemnon vive de una CNN.
- Nuestros E037 y E079b no pasaron con las mismas features, así que la expectativa es baja. Solo si sobra
  tiempo.

### I9. Más bancos externos (caro y tarde)
- **PFF Mundial 2022**: 64 partidos, **solo para evaluar** (sus términos prohíben entrenar ML).
- **SoccerNet-GAR**: tracking en Hugging Face, `OpenSportsLab/SoccerNet-GAR`, que usa jekkiechan.

Servirían de segunda opinión, pero no llegan antes del 14 nov.

## 3. Lo que confirma lo que ya hicimos

- **Validar con partidos externos degradados y entrenar fuera del reto** es lo que hacen todos los serios. Las
  puertas pre-registradas (ADR-0008) son el estándar entre los buenos.
- **La posesión controlada (ADR-0012)**: AndaAnda re-etiqueta DFL con una regla de control fechada en el
  primer toque. Mastaadmi informa con T = 1, 2 y 3 s. Ambos ven que DFL cambia de manos el doble que la GT del
  reto (lo mismo que E027).
- **Smart Tagging**: casi todos lo prueban y lo descartan.
- **Rellenar el balón con el broadcast**: aporta poco en el reto real (Mastaadmi lo admite; nosotros E023).
- **Audio**: nadie encuentra señal.
- **Oráculo**: el «quién» pesa mucho más que el D (Mastaadmi: equipo perfecto 0,32 → 0,70, D perfecto 0,45).
  Es como nuestro E009/E040.
- **Abstenerse con X no compensa**: lo confirman varios.

## 4. Limitaciones de este análisis

- Solo cuentan los repos públicos. Los equipos fuertes con repo privado (lo será hasta el final) no aparecen.
- No se ha reproducido ninguna cifra: lo que dice cada README es su afirmación.
- Se buscó en la rama con más commits de cada repo y en `main`. Puede haber trabajo en otras ramas.
- El texto de la aclaración viene de copias de otros equipos, que coinciden entre sí. **Hay que confirmarlo con
  el email original**, que debería tener Manel.
