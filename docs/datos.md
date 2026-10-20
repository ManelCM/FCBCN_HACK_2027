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
Alternativa recomendada: guardar los datos **fuera del repo y fuera de OneDrive** y apuntar a ellos con `MTAH_DATA_DIR=<ruta>`, que leen `games.py` y nuestro código.

## Inventario (Manel, 2026-10-20)
Solo nombres y tamaños; los zips pasan la comprobación de integridad (CRC).

| Fichero | Estado |
|---|---|
| tactical `atd.xml`, `atd.txt`, `homography.csv`, `smart_tagging.xml` | ✅ descargados → **tenemos todo lo imprescindible** |
| broadcast `atd.xml`, `homography.csv` | ✅ |
| broadcast `atd.txt` | ❌ falta (seguramente la parte `-009`). Es opcional |
| `tactical.mp4`, `broadcast.mp4` | ⏳ descarga en curso (`.opdownload`); hay duplicados por reintentos |

Comprobación tras descomprimir (la hace una persona): `python external/more-than-a-hack-2027/starter/load_atd.py` → debe decir `OK -- one record per video frame`.

## Entorno verificado
- Python 3.11 + `kloppy==3.18.0` + numpy/pandas/lightgbm/sklearn instalan con uv (fijar `pandas<3`).
- El scorer oficial corre y reproduce la tabla de la ventana pública con los 4 baselines.
- GPU disponible: RTX 3060 Laptop 6 GB (suficiente para un detector de balón tipo YOLO, si llegamos a ello).
