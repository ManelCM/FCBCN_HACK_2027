# Bitácora del equipo

Formato: la entrada más nueva va **arriba**.

```
## AAAA-MM-DD — Nombre
- Hecho:
- Pendiente:
- Bloqueos / preguntas para el equipo:
```

---

## 2026-10-20 (noche) — Manel
- Hecho:
  - Repo movido a `C:\dev\hack_barca` (fuera de OneDrive, sin `ç`). La copia antigua del Escritorio queda obsoleta.
  - Datos descomprimidos en `data/` y `MTAH_DATA_DIR` configurada. Alineación oficial OK (199.319/199.319). Inventario en `docs/datos.md`.
  - Entorno verificado: Python 3.11, kloppy 3.18, LightGBM y el scorer oficial.
- Pendiente:
  - Bajar de Drive la parte `-009` (broadcast `atd.txt`) y los vídeos (opcionales).
  - Crear el remoto en GitHub y hacer push.
- Bloqueos: ninguno para empezar a trabajar.

## 2026-10-20 (tarde) — Manel
- Hecho:
  - Leídas las bases oficiales (repo `metrica-sports/more-than-a-hack-2027`). `00_reto.md`, la investigación y la estrategia se han reescrito con ellas.
  - Hallazgos clave: métrica **spellF1** (segmentación, fronteras a < 0,5 s), **sin leaderboard**, **prohibido enseñar los datos a LLMs en la nube** (incluido Claude), repo de entrega **público y reproducible exacto**.
  - Nuevos ADR: 0004 (agentes ciegos a los datos) y 0005 (validación con datos externos). ADR 0001 y 0003 actualizados.
  - Añadido el repo oficial como submódulo en `external/`.
- Pendiente:
  - **Aprobar el ADR-0004 antes de que nadie descargue los datos con un agente abierto.**
  - Preguntas Q1-Q3 a la organización (issue o email).
  - Verificar los campos de DFL open data y Metrica sample-data.

## 2026-10-20 — Manel
- Hecho:
  - Investigación del reto (More Than A Hack 2027): calendario, fases, requisitos → `docs/00_reto.md`.
  - Investigación del estado del arte en posesión por frame → `docs/investigacion/01_posesion_estado_del_arte.md`.
  - Borrador de estrategia por fases → `docs/02_estrategia.md`.
  - Estructura del repo, CLAUDE.md, CONTRIBUTING, ADR 0001-0003 (en *Propuesta*).
- Pendiente:
  - Crear el repo privado en GitHub e invitar a los compañeros.
  - Descargar los datos oficiales de la fase 1 y responder Q1-Q9.
  - Revisar y aprobar (o discutir) los ADR 0001-0003.
- Bloqueos:
  - **Aún no tenemos los datos ni las bases detalladas** (métrica, formato de envío).
  - Confirmar que los 3 estamos inscritos (mínimo 3 personas para pasar de fase).
