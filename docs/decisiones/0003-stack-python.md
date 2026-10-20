# 0003 — Stack técnico: Python + uv + kloppy/DataBallPy + LightGBM

- **Estado**: Propuesta (confirmar cuando veamos el formato real de los datos)
- **Fecha**: 2026-10-20
- **Autor/a**: Manel
- **Revisado por**: ☐ ☐

## Contexto
Necesitamos un stack común para que el código de los 3 sea compatible desde el primer día.

## Opciones consideradas
1. **Python + uv**: el ecosistema de fútbol (kloppy, DataBallPy, mplsoccer, socceraction) está en Python. uv da entornos reproducibles y rápidos con un `uv.lock`.
2. **Python + conda/pip**: funciona, pero es más lento y menos reproducible.
3. **R**: buen análisis, pero un ecosistema de tracking mucho menor.

## Decisión
Python ≥ 3.11 con `uv`. Librerías base: `kloppy`, `databallpy`, `pandas`/`polars`, `numpy`, `mplsoccer`, `lightgbm`, `scikit-learn`; `torch` solo si llegamos a un modelo secuencial (L4).

## Justificación
- kloppy está recomendado por la propia Metrica para leer su formato.
- DataBallPy ya trae una posesión de equipo (Vidal-Codina) que sirve de baseline inmediato.
- LightGBM: buen rendimiento con features tabulares y poco ajuste.

## Consecuencias
- La organización fija **`kloppy==3.18.0`** para leer su EPTS y el scorer se prueba con Python 3.8 y 3.11 → usaremos **Python 3.11** (`uv python pin 3.11`) para ir sobre seguro.
- El repo oficial se incluye como **submódulo git** en `external/more-than-a-hack-2027` (commit fijado) para usar su scorer, `games.py` y sus loaders sin copiarlos. Es MIT. Clonar con `git clone --recursive`.
- La entrega debe regenerar **exactamente** el mismo fichero: `uv.lock`, semillas fijas y nada no determinista sin semilla.
