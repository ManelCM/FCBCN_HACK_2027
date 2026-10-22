# Datos: origen, estructura e inventario

Los datos **no se versionan** (ADR-0002) y **los agentes de IA no los leen** (ADR-0004).

## Origen
Carpeta de Google Drive de la organización (el enlace llega por email al inscribirse). Drive la descarga partida en zips `...Dataset-<fecha>-1-NNN.zip`, uno por fichero. La GT pública de 5 min está en el repo oficial (`external/more-than-a-hack-2027/evaluation/gt/`).

## Estructura esperada (DATA.md oficial)
```
data/
├─ tactical/                                   ← LA REFERENCIA (fila i del envío = frame i)
│  ├─ BAR-ATH_tactical.mp4                     vídeo, 199.319 frames, 25 fps (opcional para el modelo)
│  ├─ BAR-ATH_tactical_atd.xml                 metadatos del tracking   ★ imprescindible
│  ├─ BAR-ATH_tactical_atd.txt                 tracking (~196 MB)       ★ imprescindible
│  ├─ BAR-ATH_tactical_homography.csv          homografía               ★
│  └─ BAR-ATH_tactical_smart_tagging.xml       Smart Tagging            ★
└─ broadcast/                                  opcional, NO sincronizado con el táctico
   ├─ BAR-ATH_broadcast.mp4 · _atd.xml · _atd.txt · _homography.csv
```
## Configuración acordada
- Repo clonado en una **ruta ASCII fuera de OneDrive** (p. ej. `C:\dev\hack_barca`). OneDrive no debe sincronizar datos restringidos y la `ç` daba problemas de codificación en Windows.
- Datos en `<repo>/data/` (ignorado por git y bloqueado a los agentes en `.claude/settings.json`).
- Variable de entorno de usuario **`MTAH_DATA_DIR=<repo>\data`**, para que `games.py` del repo oficial encuentre los datos sin copiarlos a `external/`.

## Inventario (Manel, 2026-10-20)
Solo nombres y tamaños. Los zips pasan la comprobación CRC.

| Fichero | Estado |
|---|---|
| tactical `atd.xml`, `atd.txt`, `homography.csv`, `smart_tagging.xml` | ✅ en `data/tactical/` → **tenemos todo lo imprescindible** |
| Alineación oficial (`check_alignment_raw`) | ✅ 199.319 registros = 199.319 frames, contiguos |
| broadcast `atd.xml`, `homography.csv` | ✅ en `data/broadcast/` |
| broadcast `atd.txt` | ✅ (22 oct) movido de Descargas. `check_alignment_raw(source="broadcast")`: 164.488 registros contiguos, `agrees: True` |
| `tactical.mp4` | ✅ 02:12:52,76 = 199.319 frames a 25 fps, 1080p. Trae pista de audio, pero **muda** (−59 a −61 dB de media) |
| `broadcast.mp4` | ✅ (22 oct) movido de Descargas y renombrado (llegó como `-001.mp4`). 01:49:39,52 = 164.488 frames. **Audio real** de televisión (−26 dB de media, picos de −3 dB): sirve para ADR-0013 |

**Todos los datos del reto están completos** (22 oct). En Descargas quedan dos `.opdownload` (restos de reintentos) que se pueden borrar.

Comprobación: `python external/more-than-a-hack-2027/starter/load_atd.py` → `OK -- one record per video frame`. Duración y pistas del vídeo (solo metadatos): `ffmpeg -hide_banner -i <mp4>` con el binario portátil de `imageio-ffmpeg` (`ffprobe` no está instalado).

## Derivados y datos externos (actualizado 2026-10-21)
| Carpeta | Qué hay | ¿Se versiona? |
|---|---|---|
| `data/derived/broadcast_alignment.npz` | correspondencia táctico → broadcast (`broadcast_report`, generado el 22 oct, E023) | No (está dentro de `data/`) |
| `data/derived/logs/` | salida de los scripts sobre datos reales cuando fallan: la lee una persona, nunca un agente (ADR-0004) | No |
| `data/derived/video_ball.npz` | balón detectado en los frames de hueco (`vision.ball_detector`). Hoy solo hay una prueba: el detector genérico no pasa la puerta (E025) | Pendiente de Q4 (ADR-0010) |
| `external_data/metrica/` | Metrica sample-data, 3 partidos (descarga automática desde GitHub) | No |
| `external_data/dfl/<id>/` | DFL/IDSSE, 7 partidos. **Figshare bloquea la descarga automática: bajar a mano** `meta.xml`, `event.xml` y `tracking.xml` | No |

Lo único que falta es **DFL** (bajar a mano o con el navegador).

## Lo que sabemos del ATD (solo agregados, E005, E006 y E013)
- Balón ausente en el 40,7 % de los frames evaluados, en ~134 huecos > 4 s (mediana de 268 frames). En la ventana pública: 6,5 % vivo / 34 % muerto.
- Trayectoria del balón filtrada: residuo de 0,04 m, dv p50/p90/p99 de 1,8/5,6/11,8 m/s, velocidad p50/p90/p99 de 5,1/14,3/24,5 m/s. El balón se pierde yendo lento (p50 2 m/s).
- Speed de los jugadores presente en el 90 % de los frames; la del balón, solo en el 8,7 %.
- 46 slots (24 locales, 22 visitantes), ~22 jugadores visibles por frame (p5 = 14).

## Entorno verificado
- Python 3.11 + `kloppy==3.18.0` + numpy/pandas/lightgbm/sklearn instalan con uv (fijar `pandas<3`).
- El scorer oficial corre y reproduce la tabla de la ventana pública con los 4 baselines.
- GPU disponible: RTX 3060 Laptop 6 GB (suficiente para un detector de balón tipo YOLO, si llegamos a ello).
- Portátil de Manel: 31,4 GB de RAM (E019 se cortó con el navegador ocupando ~16 GB: cerrarlo antes de entrenar con 10 partidos). Grupo `vision` instalado: torch 2.6.0+cu124 con CUDA y ultralytics 8.4.
