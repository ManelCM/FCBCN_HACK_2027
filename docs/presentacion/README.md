# Presentación interna del equipo

Resumen visual del proyecto para el equipo: objetivo, reto y métrica, datos, cómo validamos, modelo,
resultados, análisis de fallos, decisiones y plan hasta el 17 de noviembre. **Actualizada: 22 oct 2026**
(36 diapositivas).

| fichero | qué es |
|---|---|
| `La_MasIA_presentacion_interna.pdf` | Para leer o imprimir |
| `La_MasIA_presentacion_interna.pptx` | Para PowerPoint o Google Slides. Cada diapositiva es una imagen: para editar, usa la fuente o el artifact |
| `fuente/` | Fuente editable: `deck.json` (orden y secciones) + `slides/*.html` (una diapositiva por fichero, 1920×1080) |
| `render_deck.py` | Regenera el PDF y el PPTX desde la fuente (Edge sin ventana + Pillow + python-pptx) |

- **Versión interactiva** (con notas del ponente y descarga en PPTX/PDF nativos): el artifact
  privado de Manel, https://claude.ai/artifact/VgHBPzrGZrAjzg4LByvFAE. Hay que pedirle acceso.
- **No es el documento técnico de la entrega.** Ese es `La MasIA.pdf`, cuyo borrador está en
  `docs/entrega/`.
- Solo contiene cifras agregadas y datos públicos: ningún dato del reto (ADR-0004).

## Regenerar el PDF y el PPTX

```
uv run --no-project --with python-pptx --with pillow python docs/presentacion/render_deck.py docs/presentacion/fuente <carpeta_salida>
```

Requiere Microsoft Edge (Windows). Las fuentes se cargan de Google Fonts, así que hace falta
conexión. Los iconos del artifact se sustituyen por emojis en la versión renderizada.

## Para actualizarla

1. Edita `fuente/slides/<id>.html`. Para añadir o quitar diapositivas, también `fuente/deck.json`.
2. Regenera el PDF y el PPTX con el comando de arriba y cópialos aquí.
3. Si tienes acceso al artifact, pídele al agente que lo republique.
