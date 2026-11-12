# hack_barça — More Than A Hack 2027 (BIHUB · MWCapital · Metrica Sports)

Repo del equipo (3 personas) para el hackathon **More Than A Hack 2027**. Objetivo: **ganar**, es decir, llegar a la final del 2 de marzo de 2027 en Talent Arena y ganarla.

## Calendario oficial

| Fase | Fechas | Qué hay que hacer | Cómo se pasa |
|---|---|---|---|
| Inscripción | 1 oct – **10 nov 2026** | Registrar al equipo (3-5 personas) | — |
| **Initial Challenge** | **20 oct – 17 nov 2026** (online) | Estimar la **posesión del equipo en cada frame** de un partido a partir del tracking + homografía | Ranking automático → **top 20** |
| Innovation Challenge | 26 nov 2026 – 14 ene 2027 (online) | Reto abierto de análisis/táctica sobre **10 partidos** (vídeo + tracking + eventos), con mentoría | Jurado → **top 5** |
| Final Pitch | **2 mar 2027** (presencial, Barcelona) | Pitch ante un jurado profesional | Jurado |

Más detalles en [docs/00_reto.md](docs/00_reto.md).

## Puesta en marcha

```bash
git clone --recursive https://github.com/ManelCM/FCBCN_HACK_2027.git C:/dev/hack_barca   # trae el repo oficial en external/
# Datos del reto (tras inscribirse): descomprimir en data/  → ver data/README.md
```

> ⚠️ **Antes de nada lee el ADR-0004**: las bases prohíben enseñar los datos a LLMs en la nube (Claude incluido). `.claude/settings.json` bloquea a los agentes la lectura de `data/`.

## Por dónde empezar (personas y agentes)

0. **[docs/informe/La_MasIA_informe_estado_2026-10-24.pdf](docs/informe/)**: todo lo hecho, por qué y dónde fallamos, explicado y con gráficos.
1. [CLAUDE.md](CLAUDE.md): reglas de trabajo para los agentes de IA (y para nosotros).
2. [docs/01_contexto_completo.md](docs/01_contexto_completo.md): el proyecto desde cero, al día.
3. [docs/03_plan_ganar.md](docs/03_plan_ganar.md): **estado, prioridades, reparto y cómo empezar**.
4. [docs/04_plan_investigacion.md](docs/04_plan_investigacion.md): revisión crítica, estado del arte y plan hasta el 14 nov.
5. [docs/experimentos.md](docs/experimentos.md): todos los experimentos con números (E001-E037), también los que fallaron.
6. [docs/decisiones/](docs/decisiones/): registro de decisiones (ADR 0001-0015). **Toda decisión importante está aquí.**
7. [docs/LOG.md](docs/LOG.md): bitácora del equipo.
8. [CONTRIBUTING.md](CONTRIBUTING.md), [docs/00_reto.md](docs/00_reto.md) (reto y preguntas abiertas) y [docs/datos.md](docs/datos.md).

## Cómo se genera la entrega

Un solo comando, determinista, que lee los datos del reto (lo ejecuta una persona, ADR-0004):

```bash
uv sync
uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"
```

Genera `submissions/La MasIA.csv` (199.319 filas, validado) e imprime su sha256. El modelo está
versionado en `src/hackbarca/models/artifacts/` (ADR-0007). Reserva: `--model baseline`.

Comprobación de humo con la GT pública de 5 min (no sirve para elegir modelos):

```bash
cd external/more-than-a-hack-2027/evaluation
uv run python run_evaluation.py --results ../../../submissions --out ../../../submissions/reports --gt public --no-detail
```

Banco externo v3 (10 partidos públicos, no usa datos del reto):

```bash
uv run python -m hackbarca.pipeline.tune_l3 --bench v3 --no-tags --stack --controlled 1.5 --controlled-dfl 2 --train-gt raw --jobs 2 \n    --features touch+ball+players+blocks+structure+restart
```

> **Para ponerse al día**: `docs/informe/La_MasIA_informe_estado_2026-10-24.pdf` (explicación completa con gráficos), `docs/01_contexto_completo.md` y el apartado «Estado y prioridades» + «Cómo empezar» de `docs/03_plan_ganar.md`. La rama de trabajo actual es `banco-v3`.
Estado y números en [docs/experimentos.md](docs/experimentos.md); plan en [docs/03_plan_ganar.md](docs/03_plan_ganar.md).

## Estructura

```
data/          # Datos del reto. NO se versionan (ver ADR-0002). Solo data/README.md
docs/          # Documentación, investigación, decisiones y bitácora
notebooks/     # Exploración. Lo que funcione se pasa a src/
src/hackbarca/ # io (carga, banco externo, broadcast), features, models (L1, L3), decode (Viterbi),
               # eval, pipeline (comandos), vision (detector de balón; grupo opcional 'vision')
tests/         # pytest con datos sintéticos (uv run pytest)
external/      # repo oficial del reto (submódulo)
external_data/ # datos abiertos descargados (Metrica, DFL). No se versionan
submissions/   # CSV generados. No se versionan
```
