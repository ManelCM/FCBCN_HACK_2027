# 0001 — Flujo de trabajo: GitHub + PR obligatoria + documentación en Markdown

- **Estado**: Propuesta
- **Fecha**: 2026-10-20
- **Autor/a**: Manel
- **Revisado por**: ☐ ☐

## Contexto
Somos 3 personas trabajando en paralelo, cada una con agentes de IA que escriben código y documentación. El riesgo principal es la desalineación: dos personas resolviendo lo mismo de formas incompatibles, o un agente que deshace una decisión que no conoce.

## Opciones consideradas
1. **Push directo a `main` + chat**: rápido, pero sin trazabilidad. Las decisiones se pierden en el chat y los agentes no las ven.
2. **Ramas + PR con 1 revisión + ADR/LOG en el repo**: algo más lento, pero todo queda en el repo, que es justo lo que leen los agentes.
3. **Notion/Docs externos para decidir**: buena colaboración, pero los agentes de código no lo leen de forma fiable y se desincroniza del código.

## Decisión
Opción 2. Repositorio privado en GitHub, `main` protegida (PR + 1 aprobación) y documentación en `docs/` (ADR, LOG, experimentos), con `CLAUDE.md` como contrato para los agentes.

## Justificación
- Los agentes leen el repo, no el chat; si la decisión no está en el repo, para ellos no existe.
- La revisión obligatoria hace que al menos 2 personas conozcan cada cambio.
- El coste (unos minutos por PR) es pequeño frente a 4 semanas de reto.

## Consecuencias
- Hay que revisar PRs ajenas con prioridad (objetivo: menos de 12 h).
- Las PRs triviales (typos en docs) pueden ir con *self-merge* si se avisa en el LOG.
