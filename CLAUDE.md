# Instrucciones para agentes (Claude Code u otros) y para el equipo

Trabajamos **3 personas en paralelo**, cada una con su agente. Este fichero es el contrato común para que todos los agentes vayan alineados.

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
