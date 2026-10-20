# Cómo trabajamos

## Git

- `main` está protegida: solo entra código mediante **Pull Request con al menos 1 aprobación** de otra persona del equipo.
- Ramas: `<persona>/<tema>`, por ejemplo `manel/baseline-reglas` o `ana/carga-tracking`.
- Commits pequeños, con este formato: `tipo(área): descripción`.
  - tipos: `feat`, `fix`, `exp` (experimento), `docs`, `data` (scripts de datos), `refactor`, `chore`
  - ejemplo: `exp(posesion): baseline vecino más cercano + histéresis 0.4s → 81.2% acc val`
- Antes de abrir una PR: `git fetch && git rebase origin/main`.
- La PR se rellena con la plantilla (`.github/pull_request_template.md`): qué, por qué, cómo se ha validado y qué ADR/experimento la respalda.

## Revisión

- Revisar PRs ajenas es **prioritario** frente a empezar trabajo nuevo; el objetivo es responder en menos de 12 h.
- La persona que revisa ejecuta el código, no se limita a leerlo.
- Desacuerdo técnico: se escribe en la PR. Si no se resuelve en 2 mensajes, se abre un ADR con las alternativas y decidimos los 3.

## Documentación

| Qué | Dónde | Cuándo |
|---|---|---|
| Decisión con alternativas | `docs/decisiones/NNNN-titulo.md` | Antes o junto al código que la implementa |
| Experimento y resultado | `docs/experimentos.md` | Siempre, también si sale mal |
| Qué hice hoy / bloqueos | `docs/LOG.md` | Al final de cada sesión |
| Investigación / lecturas | `docs/investigacion/` | Cuando se aprende algo reutilizable |

## Entorno

Python gestionado con `uv` (ver ADR-0003, en estado *Propuesta*). `uv sync` para instalar y `uv run python -m src....` para ejecutar.
