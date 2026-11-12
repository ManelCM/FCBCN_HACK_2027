# Estado del arte v3: la búsqueda ampliada (T1 del plan completo, 26 oct)

> Amplía `05_estado_del_arte_v2.md` (25 oct). Esta pasada cubre lo que faltaba: Kaggle, SoccerNet, tesis
> y grupos (TUM, Salford, KU Leuven), librerías abiertas (databallpy, unravelsports, Roboflow sports) y los
> trabajos clásicos de posesión y estado del juego desde el tracking.
>
> Cada técnica lleva la hipótesis para nosotros, el fallo que ataca (oráculos de E040) y **la línea del plan
> donde se prueba**. Las cifras son de sus autores y en sus datos: no son comparables con nuestro spellF1.

## Lo nuevo más útil, en una lista
1. **El estado del juego se adivina sin balón** (Lang, Wild, Isenko y Link 2022, *Scientific Reports*).
   - Solo con jugadores, en 102 partidos de la Bundesliga.
   - Acierto por frame del 92 %; F1 0,80 en parones de más de 2 s.
   - La clave es el **contexto temporal multiescala**: posiciones, velocidades y aceleraciones a ±0,4, 0,8,
     2, 4, 20, 40 y 80 s. Pesa sobre todo ±4 s, más el pasado que el futuro.
   - Fallan en los parones cortos (saques de banda y faltas rápidas: 74 % de recall) y en los tramos vivos
     muy cortos entre dos parones.
   - **Para nosotros**: en los huecos, el juego vivo sale D el 22-25 % (E040). → **F6** (`features/coach.py`,
     E071).
2. **Posesión individual y control** (Link y Hoernig 2017, *PLoS ONE*, TRACAB).
   - Un toque es un **pico de aceleración del balón ≥ 4 m/s²** junto al jugador más cercano dentro de un
     radio. La pérdida se detecta extrapolando el balón 1 s.
   - El control se clasifica con una red bayesiana a partir de la duración, la velocidad y aceleración del
     balón, la distancia y los **rivales a 0,5-5 m**.
   - F de la acción con balón: 0,88. Kappa del control: 0,67 (dos humanos: 0,72; el concepto es difuso).
   - **Para nosotros**: el ATD aplana las patadas (E013), así que los toques hay que buscarlos en **el balón
     del vídeo sin filtrar** → **U3**.
3. **La realización de la tele delata el balón muerto**.
   - Nuestra propia medida (E055-E056): con plano abierto, vivo; con primer plano o repetición, parón. AUC
     0,90 en la ventana pública.
   - No hemos encontrado ningún trabajo que lo use para la posesión. Sí hay literatura de resúmenes que usa
     el tipo de plano y las repeticiones como marcadores de acción.
   - → **U1/E057** en evaluación.
4. **Contrapresión** (Bauer y Anzer 2021, DMKD).
   - 134 features y XGBoost, AUC 0,874, con más de 20.000 transiciones etiquetadas por expertos.
   - Las claves son la presión de los más cercanos al balón en los segundos tras la pérdida y la distancia del
     bloque.
   - → **F3** (`features/coach.py`, E058).
5. **Pressing intensity** (Bekkers 2025; `unravelsports.PressingIntensity`).
   - Matriz de **tiempo hasta interceptar** de cada defensor a cada atacante, con posición, velocidad y
     reacción de 0,7 s, convertida en probabilidad con una sigmoide (σ = 0,45).
   - Funciona con kloppy y Sportec/SkillCorner, que son nuestros datos.
   - → **F4**: se puede usar la librería, o reimplementarla con numpy para no añadir dependencias.
6. **Equipo por el color de la camiseta** (SoccerNet Game State Reconstruction; Roboflow sports).
   - La tubería estándar recorta el centro del jugador, lo embebe con **SigLIP**, reduce con UMAP y separa con
     k-medias en 2 grupos.
   - Más simple: DBSCAN o k-medias en HSV del torso (Sensors 2026). El portero y el árbitro salen como grupos
     aparte.
   - → **U5**. Empezamos por color, con SigLIP como plan B.
7. **Sonido del golpeo** (Salford / FascinatE, Oldfield y Shirley).
   - El golpeo es un impulso corto y de banda ancha. Se detecta con la **tasa de cambio de energía por banda
     (dE/dt)** o con detección de *onsets* (SuperFlux), con una tolerancia de 100 ms.
   - Mejor con micrófonos de campo, pero el audio de la tele los mezcla.
   - → **A3**. Puerta: el *onset* de audio frente a los toques del balón del vídeo y del ATD.
8. **Detección de eventos desde el tracking, y una advertencia**.
   - La literatura **sobrestima** el acierto: los pases bajan a F 0,71 y los tiros a 0,65 cuando eventos y
     posiciones son independientes (estudio de validación de 2023, PMC11025972).
   - Vidal-Codina (2022) da más del 90 % de detección por categorías, pero con balón de calidad.
   - Lección: no fiarnos de umbrales tomados de artículos sin medirlos en nuestro banco degradado.
9. **Fases de juego** (Renner et al. 2026, KIT/DFB; deep embedded clustering 2026).
   - Las transiciones son lo peor clasificado (F1 0,54-0,66).
   - Para clusterizar fases hacen falta descriptores de forma del bloque y de la zona del balón.
   - → **F2**.
10. **Formaciones y roles** (Bialkowski 2014-2016; Shaw y Glickman; SoccerCPD 2022).
    - Roles por asignación húngara frame a frame, a partir de unas plantillas iniciales que se reestiman.
    - Plantillas con y sin balón por equipo (Wasserstein y clustering jerárquico).
    - Cambios de formación (SoccerCPD) a escala de minutos.
    - → **F1**.
11. **Más datos con GT de posesión y de balón muerto**.
    - **SkillCorner open data**: 20 partidos con estado del balón y poseedor por frame (**ya integrados**,
      E054).
    - PFF FC Mundial 2022: 64 partidos, por revisar.
    - El conjunto de **Lang et al.** (102 partidos con el estado anotado) no es público.
12. **Ball Action Spotting** (SoccerNet: T-DEED, 1.º en 2024; TBAS-1, 1.º en 2025, con el equipo).
    - Pases y conducciones con su equipo a ±1 s. Entrenado en tele: dominio distinto a nuestro plano táctico.
    - → **U10**, con un timebox de 1 día.
13. **Kaggle DFL Bundesliga Data Shootout** (2022): eventos (*play*, *challenge*, *throw-in*) desde vídeo de
    la Bundesliga. Hay soluciones publicadas (6.ª, Fixstars; 3.ª), basadas en modelos de vídeo 2D+tiempo.
    Útil sobre todo como referencia de U7 (saques de banda) y U10.

## Qué cambia en el plan
- **F6 entra como familia nueva**: contexto multiescala sin balón, según Lang et al. Es barata y ataca
  directamente los huecos.
- **U3 se basa en Link y Hoernig**: pico de aceleración ≥ 4 m/s² más cercanía, sobre el balón del vídeo.
  Requiere pasar el detector de balón por **todos** los frames, no solo por los huecos: una pasada de GPU de
  noche.
- **A3 usa dE/dt por banda y *onsets***, no las clases de PANNs.
- **F4 puede reutilizar `unravelsports`**: comprobar la licencia y el determinismo antes de añadirlo; si no,
  numpy propio.

## Fuentes
- Lang et al. 2022: https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/
- Link y Hoernig 2017: https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/
- Bauer y Anzer 2021: https://link.springer.com/article/10.1007/s10618-021-00763-7
- Pressing Intensity: https://arxiv.org/abs/2501.04712 · https://unravelsports.readthedocs.io/en/latest/api/generated/unravel.soccer.PressingIntensity.html
- SoccerNet GSR: https://arxiv.org/abs/2404.11335 · SoccerNet 2025: https://arxiv.org/abs/2508.19182
- Roboflow (SigLIP + UMAP + k-medias): https://blog.roboflow.com/identify-basketball-players/
- Equipo por color (Sensors 2026): https://www.mdpi.com/1424-8220/26/4/1252
- Salford / FascinatE (golpeo): https://salford-repository.worktribe.com/output/1332765/application-of-object-based-audio-for-automated-mixing-of-live-football-broadcast
- Validación de detección de eventos: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025972/
- Vidal-Codina 2022: https://arxiv.org/abs/2202.00804 · Slicing and dicing soccer: https://arxiv.org/abs/2004.04147
- Fases con GNN: https://arxiv.org/abs/2610.11571
- databallpy (posesión): https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html
- DFL Bundesliga Data Shootout: https://bundesliga.com/en/aws-bundesliga-data-shootout
- Audio y comentario para resúmenes: https://repositorio.ucp.pt/entities/publication/6981957f-9c58-4f84-85c2-4f3353bdd987 · https://arxiv.org/abs/2405.07354
