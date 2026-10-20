# 0002 — Los datos del reto no se suben a git

- **Estado**: Propuesta
- **Fecha**: 2026-10-20
- **Autor/a**: Manel
- **Revisado por**: ☐ ☐

## Contexto
Los datos (vídeo, tracking y eventos de partidos del FC Barcelona) los proporcionan BIHUB y Metrica. Son pesados (el vídeo ocupa GB) y, casi seguro, tienen restricciones de uso.

## Opciones consideradas
1. **Subirlos al repo / Git LFS**: cómodo, pero viola probablemente la licencia y GitHub limita el tamaño (LFS de pago a partir de 1 GB).
2. **Que cada persona los descargue de la fuente oficial en `data/raw/`, y compartir solo lo que producimos (etiquetas manuales) por Drive**.

## Decisión
Opción 2. `data/` en `.gitignore`, salvo `data/README.md`.

## Justificación
Cumplimiento de las condiciones de uso (pendiente de leerlas, Q9) y un repo ligero y rápido de clonar.

## Consecuencias
- El código debe cargar desde `data/raw/` con rutas relativas y estructura documentada en `data/README.md`.
- Todo dato derivado debe poder regenerarse con un script de `src/`.
