# 0004 — Política de uso de agentes de IA (Claude Code) con los datos del reto

- **Estado**: Propuesta — **URGENTE, revisar antes de descargar los datos**
- **Fecha**: 2026-10-20
- **Autor/a**: Manel
- **Revisado por**: ☐ ☐

## Contexto
Los T&C (citados en FAQ.md y OVERVIEW.md del repo oficial) dicen:
> *participants shall not upload, disclose or otherwise make available to any third-party hosted or cloud-based LLM any videos, datasets, confidential information or other restricted materials provided in connection with the Competition, **or any extracts thereof***.

Claude Code es un LLM alojado en la nube: **todo lo que un agente lee (ficheros, salida de comandos, imágenes) se envía a Anthropic.** Si un agente hace `head` del ATD, imprime un DataFrame, abre un frame del vídeo o lee el XML de Smart Tagging, incumplimos las reglas y nos arriesgamos a la **descalificación**.

## Opciones consideradas
1. **No usar agentes**: seguro, pero perdemos mucha velocidad.
2. **Agentes "ciegos a los datos"**: los agentes escriben código, documentación y tests con **datos sintéticos o datos públicos externos**. Solo las personas ejecutan el código sobre los datos del reto, y lo que vuelve al agente es, como mucho, una **métrica agregada** del scorer.
3. **LLM local** (p. ej. Ollama) para todo lo que toque datos: permitido, pero es más lento y peor.

## Decisión (propuesta)
Opción 2, con opción 3 para casos puntuales:
- `data/` del reto está **fuera del alcance de los agentes**. Están **prohibidos** `cat`/`head`/`Read`/`Grep` sobre `data/`, mostrar DataFrames o filas, mostrar imágenes o frames, y leer los CSV de predicciones (también son derivados frame a frame).
- Los scripts que ejecute un agente sobre datos reales **solo pueden imprimir métricas agregadas** (spellF1, macroF1, número de posesiones). Hasta que la organización responda a Q2 de `00_reto.md`, **es preferible que lo ejecute una persona** y pase al agente solo el número.
- Para desarrollar y depurar, los agentes usan **datos públicos** (Metrica sample-data, DFL open data) y la **GT pública de 5 min**, que es pública en GitHub y no es un extracto de los datos restringidos.
- Visualizaciones de los datos del reto: las mira una persona, en local, nunca un agente.

## Justificación
Cumplir las reglas es condición necesaria para ganar. Se puede trabajar en el 90 % del código (cargadores, features, HMM, evaluación, pipeline de datos externos) sin que un agente vea los datos restringidos, porque el formato está documentado públicamente en DATA.md y los datos externos tienen la misma forma.

## Consecuencias
- Hay que añadir esta regla a `CLAUDE.md` (hecho) y, si se puede, un *hook* de Claude Code que bloquee las lecturas sobre `data/`.
- Enviar a la organización la pregunta Q2 (¿cuentan las métricas agregadas como extracto?).
- El desarrollo debe tener un **modo sintético o externo** desde el primer día (los tests no dependen de `data/`).
