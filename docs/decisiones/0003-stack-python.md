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
Fijar la versión de Python que funcione con todas las dependencias (ojo: algunas librerías pueden no tener *wheels* para 3.14 todavía; probablemente 3.12).
