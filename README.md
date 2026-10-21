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

1. [CLAUDE.md](CLAUDE.md): reglas de trabajo para los agentes de IA (y para nosotros).
2. [CONTRIBUTING.md](CONTRIBUTING.md): flujo de git, revisiones y cómo documentar.
3. [docs/00_reto.md](docs/00_reto.md): qué se pide, qué sabemos y qué falta por saber.
4. [docs/investigacion/01_posesion_estado_del_arte.md](docs/investigacion/01_posesion_estado_del_arte.md): investigación técnica de la fase 1.
5. [docs/02_estrategia.md](docs/02_estrategia.md): plan para ganar, por fases.
6. [docs/decisiones/](docs/decisiones/): registro de decisiones (ADR). **Toda decisión importante está aquí.**
7. [docs/LOG.md](docs/LOG.md): bitácora del equipo (qué hizo cada persona y cuándo).

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

Banco externo (no usa datos del reto): `uv run python -m hackbarca.pipeline.tune_l3 --features 3 --no-tags`.
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
