# U10: acciones con el balón y su equipo desde el broadcast (T-DEED preentrenado)

Estado (27 oct): **preparado, sin ejecutar**. Falta el permiso para ejecutar código y pesos de terceros.

## Qué es
- **Modelo.** Es el baseline oficial del reto *SoccerNet Team Ball Action Spotting 2025*: T-DEED, con RegNetY-002 + GSF y SGP-Mixer, y una cabeza de equipo sigmoide.
  - Detecta 12 acciones con el balón: pase, conducción, cabezazo, pase alto, fuera, centro, saque de banda, tiro, bloqueo, entrada con éxito, falta y gol.
  - Para cada acción dice si la hace el equipo de la **izquierda** o de la **derecha** del vídeo.
- **Uso en este proyecto:**
  - las acciones con equipo dan quién tiene el balón en cada instante;
  - fuera, saque de banda y falta marcan D;
  - una secuencia de pases de un equipo seguida de una acción del otro marca un cambio de posesión.
- **Por qué sobre el broadcast y no sobre el vídeo táctico.** El modelo se entrenó con vídeo de televisión: 25 fps reducidos a 796×448, tomando 1 de cada 2 frames. El broadcast del reto es el mismo dominio y ya está alineado con el táctico por partes (E023).

## Origen y licencia (fuera de git)
- **Código:** `external_data/tdeed/team`, clonado de https://github.com/SoccerNet/sn-teamspotting. Licencia GPL-3.0. Solo se importa en tiempo de ejecución, no se copia a `src/`.
- **Pesos:** `external_data/tdeed/ckpt/SoccerNetTeamBall/SoccerNetBall_baseline/checkpoint_best.pt`, 48 MB. Vienen del Drive enlazado en el README de T-DEED (https://github.com/arturxe2/T-DEED). Se cargan con `weights_only=True`.
- **Dependencia nueva:** `timm` (grupo `vision`).

## Cómo se ejecuta
```
scripts/run_aggregate.sh ball_actions uv run --group vision python -m hackbarca.vision.ball_action_pass
```
- Procesa solo las partes puntuadas: frames del broadcast según `broadcast_alignment.npz`.
- Usa clips de 100 pasos con avance de 50, y de cada clip guarda la mitad central.
- Escribe en `data/derived/ball_actions/half<k>.npz`: el frame del broadcast y 25 probabilidades.
- Solo imprime progreso y recuentos.

## Puerta prevista (E097, pseudoetiquetas de la entrega, ADR-0019)
- **Equipo.** Acuerdo del equipo de las acciones con la predicción segura (pmax ≥ 0,9) en juego vivo ≥ 0,85, con cobertura ≥ 10 % de los frames dudosos o de los huecos del ATD.
  - El lado izquierda/derecha → local/visitante se fija por parte (1 bit), y se comprueba que cambia en el descanso.
- **D.** Fuera, saque de banda y falta seguidos de D predicho a ≤ 3 s, con lift ≥ 1,5 sobre la base.
- **Si pasa:** simular la señal en el banco con el acierto y la cobertura medidos (patrón de `io/video_conductor.py`) y pasar el protocolo.
