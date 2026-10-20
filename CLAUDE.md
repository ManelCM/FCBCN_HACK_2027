# Instrucciones para agentes (Claude Code u otros) y para el equipo

Trabajamos **3 personas en paralelo**, cada una con su agente. Este fichero es el contrato común para que todos los agentes vayan alineados.

## ⛔ REGLA Nº 1 — Los agentes NO ven los datos del reto (ADR-0004)

Las bases prohíben pasar a un LLM en la nube los datos del reto **o extractos de ellos**, y este agente es un LLM en la nube. Por tanto:
- **Nunca** leas, hagas `cat`/`head`/`grep`, abras ni listes el contenido de ficheros en `data/` (ATD, XML, homografía, vídeo, Smart Tagging) ni de predicciones frame a frame (`submissions/`, `evaluation/results/*.csv` propios).
- **Nunca** imprimas filas, DataFrames, arrays ni imágenes que vengan de esos datos. Los scripts que ejecutes sobre datos reales solo pueden sacar **métricas agregadas**, y mejor que los ejecute una persona.
- Para desarrollar y probar, usa la documentación pública (DATA.md del repo oficial), **datos sintéticos**, **datos abiertos externos** (Metrica sample-data, DFL) y la **GT pública de 5 min** del repo oficial.
- Si una tarea parece necesitar ver los datos, **para y pide a la persona** que lo mire ella.

## Fuente de verdad del reto
Repo oficial: https://github.com/metrica-sports/more-than-a-hack-2027. Resumen en `docs/00_reto.md`. La métrica es **spellF1** (segmentación en posesiones), no la accuracy: lee `docs/investigacion/01_posesion_estado_del_arte.md` §1 antes de proponer cualquier modelo.

## Antes de hacer nada

1. `git pull` y lee `docs/LOG.md` (últimas entradas) y `docs/decisiones/` (ADRs en estado *Aceptada* o *Propuesta*).
2. Revisa los commits nuevos de los compañeros: `git log --oneline -20` y `git diff <tu_último_commit_visto>..origin/main`. Si algo choca con una decisión aceptada, avisa (issue o comentario en la PR); no lo "arregles" en silencio.
3. Si tu tarea depende de algo que aún no se sabe (formato de datos, métrica...), mira la sección "Preguntas abiertas" de `docs/00_reto.md`.

## Reglas

- **Idioma**: documentación y commits en castellano. Identificadores de código en inglés.
- **Nada de datos en git**: `data/` está en `.gitignore` (ADR-0002). Nunca subas vídeo, tracking ni submissions con datos del reto.
- **Toda decisión no trivial genera un ADR** en `docs/decisiones/` (plantilla: `0000-plantilla.md`). Una decisión es no trivial si otro compañero podría haber elegido otra cosa razonable: modelo, definición de posesión, features, stack, formato o qué se envía al leaderboard.
- **Todo experimento se registra** con lo que se probó, el resultado (número) y la conclusión, en `docs/experimentos.md`. Un experimento fallido también es información.
- **Cada sesión de trabajo deja una entrada en `docs/LOG.md`** con fecha, autor, qué se hizo, qué queda y qué bloquea.
- **Nunca trabajes en `main` directamente**: rama + PR + revisión de otra persona (ver CONTRIBUTING.md).
- **Justifica con datos**: si afirmas "X funciona mejor", acompáñalo de la métrica, el partido o el segmento y el commit que lo reproduce.
- **Reproducibilidad**: el código de `src/` se ejecuta con `uv run ...`; las semillas son fijas y las rutas son relativas a la raíz del repo.

## Revisión de cambios ajenos (obligatorio para agentes)

Cuando revises una PR o commits de un compañero, comprueba:
1. ¿Respeta los ADR aceptados? Si no, ¿hay un ADR nuevo que lo justifique?
2. ¿Hay entrada en LOG/experimentos con resultados?
3. ¿El código corre (`uv run`) y no rompe la evaluación local?
4. ¿Hay riesgo de *leakage*, por ejemplo usar datos del partido de test para ajustar umbrales sin decirlo?
Deja el feedback como comentario en la PR, con propuestas concretas.
