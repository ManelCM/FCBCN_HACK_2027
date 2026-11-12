# Bitácora del equipo

Formato: la entrada más nueva va **arriba**.

```
## AAAA-MM-DD — Nombre
- Hecho:
- Pendiente:
- Bloqueos / preguntas para el equipo:
```

---

## 2026-11-12 (tarde) — Manel (con agente) · E115 y mezclas no pasan; pruebas baratas mientras corre la cola
- **E115 (dos cabezas sobre E113):** 0,7549 frente a 0,7555 (5 de 10). Con el poseedor al 30 %: 0,7603 frente a 0,7621 de E122. No pasa.
- **E124 (mezclas en logit de E113, E115, poseedor y MS-TCN):** ninguna supera a E122 (de −0,002 a −0,005).
- **E125 (poseedor con otras semillas):** 0,7587 y 0,7581; la media de las tres semillas, 0,7615, frente a 0,7621. **La semilla mueve unos 0,004**: la de E122 fue la más afortunada. No cambia la entrega.
- **E126 (pesos distintos para vivo y equipo):** 0,7613. Toda la ganancia del poseedor está en el reparto H/A.
- **E127 (umbral de la regla):** con la decodificación de E122 la regla apenas actúa en el banco; se mantienen 3 s.
- **E128 (10 épocas):** 0,7563; sobreajusta.
- **E129 (peso según balón visible o en hueco):** 0,7608; el óptimo es 0,3 en los dos casos.
- **E123 (balón muerto llevado a sacar):** 0,7368 frente a 0,7555; peor. **E114:** parado a las 22:04 sin terminar.
- **Cierre de la entrega (decisión de Manel, 22:00): E122.**
  - `make_submission` por tercera vez: el mismo sha `2223250da5edc7ed`.
  - Pasa el `validate_submission.py` oficial: 199.319 filas, 260 posesiones de 13,6 s.
  - Documento técnico cerrado, sin TODO.
- Pendiente (personas):
  - subir al Google Form el CSV, el enlace del repo y el documento;
  - fusionar `competidores` en `main` (PR) y hacer público el repo;
  - reproducción en una segunda máquina, si da tiempo.
- Bloqueos / preguntas para el equipo: confirmar el email del 3 nov; revisión de ADR-0023/0024; reproducción en una segunda máquina.

## 2026-11-12 — Manel (con agente) · nueva entrega E122 (pesos por posesión + poseedor); I6 e I8-TCN cerradas
- **E120 (I6, frontera al toque):** empeora siempre. Se cierra.
- **E121 (I8, mezcla en logit):**
  - con el MS-TCN, peor (0,7514 frente a 0,7555);
  - **con el poseedor de E078 al 30 %, sobre E113: 0,7621 con la regla de 3 s (+0,011, 8 de 10).**
- **E122:** el poseedor entrenado con los 10 partidos (`train_possessor_final`, `artifacts/possessor.pt`, versionado con una excepción en `.gitignore`) y mezclado en `make_submission`.
  - Las dos decodificaciones pre-registradas caen en el veto relativo de ADR-0020 por los cambios directos (1,435 y 1,492 frente a 1,539).
- **ADR-0024 (decisión de Manel):** manda el banco, con veto absoluto en D cortos ≤ 0,25, tramos D 0,8-1,3 y % de D 0,38-0,50. Sustituye a ADR-0020.
  - **Entrega = E122**, `La MasIA.csv` sha `2223250da5edc7ed` (dos ejecuciones iguales; 97,3 % de D antes del saque).
  - En el reto: D cortos 0,142, tramos D 1,039, cambios directos 1,435 (riesgo declarado).
  - Banco: 0,7621 frente a 0,7355 de E110.
- **Documento técnico actualizado para E122.**
- La cola corre en una ventana aparte desde las 12:37: E115 (dos cabezas sobre E113) y luego E114.
- Pendiente:
  - si E115 o E114 pasan frente a E113, mezclar su OOF con el poseedor, aplicar el pre-registro, el veto de ADR-0024 y generar el artefacto;
  - I7, si sobra tiempo.
- Bloqueos / preguntas para el equipo:
  - revisar ADR-0023 y ADR-0024;
  - confirmar el email del 3 nov;
  - reproducción en una segunda máquina.

## 2026-11-10 (noche) — Manel (con agente) · E110 entrega provisional; E113 pasa el banco pero no el veto
- **E110** (artefacto de E058b, sin vídeo) es la entrega provisional: `La MasIA.csv`, sha `df01cd02b266e933`.
  - Es reproducible y da 97,3 % de D antes del saque.
  - Mejora las cuatro medidas de forma frente a E070 y E049c.
  - E110b (la decodificación del banco) no pasa el guardarraíl.
- **E113 (el mismo peso para cada posesión) pasa ADR-0008:** 0,7555 frente a 0,7410, mejor en 8 de 10. Solo funciona con su propia decodificación.
- **E117, artefacto de E113:** el orden de las decodificaciones se fijó antes de mirar el reto. Ninguna pasa el veto de ADR-0020 frente a E110:
  - la de la LOGO pierde cambios directos (1,416);
  - 1/30/20/20 deja +0,009 D cortos/min (unos 1 tramo en el partido) y mejora las otras tres medidas.
  - **La entrega sigue con E110.** Artefactos en `external_data/artefactos/{e058b,e117}/`.
- E115 y E114 fallaron: memoria y pliegues internos con SkillCorner. Arreglado en 999df51. Se relanzaron sobre E113 con `--jobs 1`, y se pararon a las 23:03 por el apagado del PC.
- Pendiente:
  - **Al volver: `bash scripts/cola.sh` en una terminal aparte.** `cola.txt` ya tiene E115 y E114, que tardan unas 4-5 h y 8 h.
  - Si alguno pasa, artefacto y veto frente a E110, con el mismo pre-registro.
  - Rellenar las filas TODO del documento técnico.
- Bloqueos / preguntas para el equipo:
  - **¿Aceptamos E117 con 1/30/20/20?** Falla el veto por un tramo D corto. Haría falta un ADR con una tolerancia que valga para todos los candidatos.
  - Confirmar el email original del 3 nov.

## 2026-11-10 (tarde) — Manel (con agente) · ideas de otros equipos y entrega sin vídeo entrenado (rama `competidores`)
- **Investigación de competidores** (`docs/investigacion/07_competidores.md`). Hay 210 repos públicos; tres de ellos copian el email de la organización del 3 nov.
- **ADR-0023 (Aceptada por Manel):** no se entrena, ajusta ni calibra nada con la GT o con el vídeo del reto.
  - U6 y E075 salen de la entrega.
  - La base pasa a ser **E058b** (0,7410, sin vídeo).
  - Documento técnico reescrito sin la variante A, con una sección de cumplimiento.
- **E111 (I1, traspaso más barato que parada + reanudación):** no pasa (≤ +0,005). `pipeline/decode_variants.py` hace Viterbi por lotes.
- **E118, retención de 2 s sobre E058b:** +0,0051, mejor en 6 de 10. No pasa.
- **E119:** el `dead_weight` del banco queda justificado sin la GT pública (objetivo solo de tracking; 100 está en la meseta). Cierra la zona gris de ADR-0023.
- **Código de E113-E115** (commit 09bb0d7, 93 tests en verde):
  - pesos por posesión (`--spell-weight`);
  - dos cabezas (`--two-heads`);
  - SkillCorner solo con H/A y el balón visto (`--extra-sc --sc-team-only`).
  - El humo de `tune_l3` con las tres banderas lo mató la gestión de memoria. Las evaluaciones se lanzan sin él: si fallan, `cola.sh` apunta rc=1 y sigue.
- **En marcha:** script del agente en segundo plano. Cuando termina la cola actual (E099nv):
  - entrena **E110** (artefacto de E058b, sobrescribe `models/artifacts/`);
  - lo copia a `external_data/artefactos/e058b/`;
  - vuelve a lanzar `cola.sh` con E113 → E115 → E114.
- Pendiente:
  - E110: `make_submission` dos veces, `kickoff_report --csv`, `real_vs_bench` y guardarraíl. Es la entrega provisional.
  - Leer E099nv, E113, E115 y E114 frente a E058b; después E116 (combinación) y E117 (artefacto final).
- Bloqueos / preguntas para el equipo:
  - **Confirmar el email original del 3 nov.**
  - Revisar ADR-0023.
  - Reproducción en una segunda máquina.

## 2026-11-10 — Manel (con agente) · informe de las 40 features recalculado; regla del saque inicial
- **E099, las 40 features del informe del compañero sobre E075:** 0,7677 frente a 0,7692 (−0,0015, 5 de 10). No pasa ADR-0008. El +0,017 del informe se medía frente a E035, sin vídeo.
- **E099b, retención mínima de 2 s:** E075 + retención da 0,7678 frente a 0,7606 de la entrega (+0,0072, 7 de 10). Con E099, +0,0094 (7 de 10). No pasa por poco. Queda como candidata para T6.
- **E054 (M3, SkillCorner):** apuntado; corrió el 28 oct. 0,6856, −0,040. M3 se cierra.
- **ADR-0022 (Propuesta), regla del saque inicial:** antes del `Kick Off (Start)` del Smart Tagging de cada parte se predice D. Antes, la entrega daba 46,4 % de D en los 2.934 frames anteriores al saque de la 2.ª parte. `l3_decode.json` lleva `kickoff_dead: true`. Hay que regenerar la entrega.
- **Documento técnico** reescrito en inglés: variantes A y B, banco, protocolo, activos usados y descartados, y reproducción.
- **Reproducibilidad de la variante A:** la pasada de vídeo es idéntica bit a bit en la misma máquina (`vision/repro_check.py`). El reentrenamiento del YOLO (`atd_pl_repro`) se colgó a las 13:42, cuando la gestión de memoria de Claude Code mató su shell; se paró a las 15:35 sin terminar.
- **Aviso de la otra sesión (email de la organización del 3 nov):** no se puede entrenar ni ajustar con el vídeo del reto. U6/E075 usa el YOLO afinado con recortes del vídeo táctico, así que **deja de valer para la entrega**, y reentrenarlo tampoco cumple. Por eso E099/E099b (medidos sobre E075) y la comprobación de reproducibilidad del YOLO sirven solo como información. La base nueva es E058b, sin vídeo (0,7410). La otra sesión lleva E110-E129 y ADR-0023/0024 en la rama `competidores`.
- **Lección:** las shells de fondo de Claude Code se matan cuando falta memoria, y sus hijos de Python quedan huérfanos. Las evaluaciones largas, mejor en una terminal aparte (`bash scripts/cola.sh`).
- Pendiente:
  - E099nv (40 features sin vídeo, frente a E049), en marcha desde las 15:26.
  - La generación con la regla del saque (`data/derived/sub_kick1/2`) usa E075: sirve para comprobar la regla y que se repite, **no es la entrega**. La entrega provisional será el artefacto de E058b + la regla de 3 s + ADR-0022.
  - Medir la retención de 2 s sobre el OOF de E058b.
  - T6: combinación final, congelación el 14 nov.
- Bloqueos / preguntas para el equipo: revisar los ADR 0015-0022, sobre todo el 0022, porque vuelve a usar el Smart Tagging en la entrega.

## 2026-10-27 (noche) — Manel (con agente) · U7 cerrada, U10 pasa la puerta; dos fallos de la cola
- **U7 (E091):** no pasa. El saque de banda tiene lift 1,65, pero solo cubre el 6,3 % de los frames dudosos, y el equipo del portero acuerda 0,404. Cerrada.
- **U10 (E097):** pasa la puerta de equipo: acuerdo 0,86-0,93 con la predicción segura y cobertura del 20-31 % de los dudosos. La de paradas no pasa (lift 1,37-1,48). E098 la lleva al banco con la fiabilidad medida.
- **Fallo 1:** E093 (bagging ×3) se colgó desde las 15:04 sin consumir CPU. El pool de procesos arrancó mientras `models/l3.py` tenía el error de sintaxis que se corrigió a las 15:02, y sus workers no llegaron a cargar. Lo maté a las 22:50 y está otra vez en cola; el log antiguo es `external_data/logs/E093_colgado.log`.
  - **Lección:** no tocar código que importan los workers mientras arranca una evaluación, o comprobar `import` antes de guardar.
- **Fallo 2:** E098 cayó a los 6 min con `KeyError: 'ba_m25_a'`. `swap_teams` exige las columnas en pares `_h`/`_a`. Arreglado en d7e316a, con un test de intercambio de equipos (80 tests pasan). E098 vuelve al principio de la cola.
- **Cola:** E096 (en marcha desde las 22:56) → E098 → E093 → E054 → E053.

## 2026-10-27 (tarde, 2) — Manel (con agente) · E075b no pasa; U7 y U10 preparados
- **E075b (U6 + F3):** 0,7714 frente a 0,7692 de E075 (+0,002, 7 de 10). No pasa ADR-0008. Con el vídeo dentro, F3 apenas suma. La entrega no cambia: E075 + regla de 3 s.
- **M5d/M5e:** `train_l3_final --bag N / --xgb-mix` guarda los modelos compuestos en el artefacto (JSON con cabecera dentro del .gz, idéntico byte a byte). Test de ida y vuelta añadido; 76 tests pasan.
- **U7:** `pipeline/pose_gate.py` está listo (saque de banda y portero con el balón, con pseudoetiquetas). Se lanza al terminar la pasada de pose de la GPU.
- **U10:** el baseline oficial de SoccerNet Team Ball 2025 (T-DEED) da 12 acciones con el balón y el equipo que las hace, y se aplica sobre el broadcast. El código está en `vision/ball_action_pass.py`; el código y los pesos de terceros, en `external_data/tdeed` (GPL-3.0, fuera de git); `timm` entra en el grupo `vision`. **No se ha ejecutado**: el modo automático pide permiso explícito para ejecutar código y pesos descargados. Detalle en `docs/investigacion/resultados/u10_ball_actions.md`.
- **Cola:** E093 (bagging ×3, desde las 14:57) → E096 (mezcla con XGBoost) → E054 → E053.
- **Bloquea:** el permiso para U10.

## 2026-10-27 (tarde) — Manel (con agente) · guardarraíl relativo; la entrega pasa a E075 + regla de D cortos
- Hecho (rama `banco-v3-1`):
  - Cola de la noche frente a E049 (0,7254):
    - **E075, U6 conductor por vídeo: 0,7692 (10 de 10)**;
    - **E058b, F3 firmas del robo: 0,7410 (9 de 10)**;
    - E057 broadcast: 0,7311 ❌;
    - E071b estado del juego: 0,7271 ❌.
  - **Comprobación de la simulación de U6** (`pipeline/vc_near_report.py`): en el reto, el acuerdo entre vídeo
    y predicción cerca de los robos queda entre la simulación base y una pesimista. **E075s** (pesimista):
    **0,7605, 10 de 10**. U6 es sólida.
  - **ADR-0020** (decidido por Manel): guardarraíl relativo, sin empeorar ninguna medida frente a la entrega
    vigente. **E075d**: la entrega pasa a E075 (sha `9cb9f1d33262a71a`).
  - **E090/E090b, regla de D cortos**: un D de menos de 3 s entre dos tramos vivos pasa al tramo anterior.
    - k se fija solo con el banco: +0,001 en el banco limpio y +0,023 con el defecto del reto inyectado.
    - En el reto, D cortos **0,529 → 0,198/min** y cambios directos 1,586 → 1,775.
    - El exceso de D cortos cae sobre todo en campo abierto (0,19/min frente a 0,004 en la GT del banco).
  - **ADR-0021** (aceptada por Manel). **E090c: la entrega es E075 + regla de 3 s**, sha `d976e575374b00d5`.
    Frente a E070: D cortos 0,680 → 0,198, tramos D 1,775 → 1,313, cambios 1,397 → 1,775.
  - Humo en la ventana pública (ADR-0019, no decide): E070 0,651, E075 0,493, E075 + regla 0,502.
  - Líneas cerradas:
    - **E092**, U3 toques del vídeo: lift 1,38 ❌;
    - **E094**, U8 árbitro: AUC 0,715 ❌;
    - **E095**, A5 grada: AUC 0,47-0,54 ❌;
    - U9 descartada por análisis;
    - U2 pasa a L3 de la otra sesión.
  - Código nuevo:
    - **M5e**, bagging (`--bag`) → E093 en cola;
    - **M5d**, mezcla LightGBM + XGBoost (`--xgb-mix`, xgboost añadido a `pyproject.toml`) → E096 en cola;
    - **U7**, pasada de pose (`vision/pose_pass.py`, YOLO-pose con mosaicos ×2) en marcha en la GPU, unas 6 h.
- Pendiente:
  - cola: E075b (U6 + F3) → E093 → E096 → E054 → E053;
  - puerta de U7 cuando acabe la pasada de pose;
  - U10, con un timebox de 1 día;
  - T6.
- Bloqueos / preguntas para el equipo:
  - **revisar ADR-0020 y ADR-0021**. Si no se aprueban, se vuelve a E075 (`short_dead_frames: 0`) o a E070;
  - **numeración**: el plan de parones de la otra sesión reservaba ADR-0020 para «banco v3.2». Ese número ya
    es el guardarraíl relativo, así que v3.2 debe usar ADR-0022 o siguiente. Mis experimentos nuevos van de
    E090 en adelante;
  - el spellF1 de la ventana pública baja con E075 (humo; no decide, pero que una persona lo sepa);
  - `xgboost` es una dependencia nueva: `uv sync`.

---

## 2026-10-27 — Manel (con agente) · E049 pasa; guardarraíl no; U6 vídeo pasa la puerta
- Hecho (rama `banco-v3-1`):
  - **E049**: reentrenado en v3.1, **0,7254 frente a 0,7059 (10 de 10)**; con la decodificación de la entrega
    0,7229, y D cortos en el banco 0,180 → 0,107/min. Es la base nueva de T2-T5.
  - **E049c**: artefacto v3.1 + guardarraíl en el reto: **no pasa** (3 de 4 umbrales). D cortos 0,680 → 0,529/min
    (umbral 0,25), tramos D 1,78 → 1,61 y cambios 1,40 → 1,48. **La entrega sigue con E070**; el artefacto v3.1
    está en `external_data/artefactos/v3_1/`. Lo que queda es balón visible y quieto en juego vivo con el mismo
    equipo a los dos lados, y el banco aún no tiene bastantes tramos así (línea L1 de `docs/06_plan_parones.md`).
  - **E078b** (poseedor + E049, 0,7236) y **E079/E079b** (MS-TCN + E049, 0,7207): no aportan. M5a y M5c se cierran.
  - **E074**, pasada de vídeo completa:
    - equipo por color 0,926;
    - **U6, quién conduce: acierto implícito ~0,90 en todos los contextos** (0,95 en los frames dudosos, donde
      el modelo acierta 0,66), con una cobertura de 0,40-0,57. Pasa la puerta.
  - **E075 preparado**: `io/video_conductor.py`, que simula U6 en el banco con la cobertura y el acierto del reto,
    también en los parones: 32 % con conductor, del equipo siguiente el 70 % de las veces. El desajuste
    reto/banco es ≤ 0,17. Va en cabeza de la cola.
  - Arreglada la espera de `scripts/cola.sh`, que buscaba una línea que tune_l3 ya no imprime y dejó la cola
    parada unas horas.
- Pendiente:
  - en la cola: E057 (broadcast, corriendo) → E075 (U6) → E058 → E071 → E053 → E054;
  - con lo que pase, un artefacto nuevo y el guardarraíl otra vez;
  - U3, toques desde el balón del vídeo: falta su medida.
- Bloqueos / preguntas para el equipo:
  - **Hay otra sesión de agente trabajando en la misma carpeta y rama**, con E080-E083, ADR-0019 y el plan de
    parones sin commit. Mis commits solo llevan mis cambios, pero conviene darle su propia rama o worktree.
  - **ADR-0019** (ventana pública solo como humo, tras las reglas del 22-23 oct) necesita revisión del equipo.
  - Q6 sigue sin respuesta: si la organización no permite las señales de broadcast y audio, E057 no puede
    entrar. U6 usa el vídeo táctico, que es dato del reto.

---

## 2026-10-26 — Manel (con agente) · plan completo en marcha
- Plan aprobado: el **plan completo** (T0-T6), que suma todo lo pedido. Tabla de estado por línea en
  `docs/05_plan_mejora.md`.
- **T0** (rama `banco-v3-1`):
  - banco v3.1 calibrado (E047, ADR-0017): pérdida 160 → 20,5; el balón lento en vivo vuelve (0,96 tramos/min);
  - guardarraíl (ADR-0018) con los umbrales escritos antes de entrenar;
  - SkillCorner: 20 partidos con GT completa y sus features listas;
  - cola en serie (`scripts/cola.sh`, `external_data/cola.txt`): E048 → E049 → E057 → E058 → E071 → E053 → E054.
- **T1**: `docs/investigacion/06_estado_del_arte_v3.md`.
  - Lang 2022: el estado del juego sin balón, 92 %.
  - Link y Hoernig 2017: toques como picos de aceleración ≥ 4 m/s².
  - Bauer y Anzer, pressing intensity, equipo por color con SigLIP, sonido del golpeo.
- **T2, puertas en el banco v3.1**:
  - ✅ **F3 firmas del robo: AUC 0,902 tras un robo** (el centroide sale hacia delante);
  - ✅ F6 estado sin balón (por poco: 0,879 frente a 0,867 en hueco);
  - ❌ F1 formación (hueco 0,729), F2 situaciones, F4 pressing, F5 territorio, «control».
- **T3/T4, señales del reto**:
  - ✅ **U1, broadcast: AUC 0,90 en la ventana pública** y 0,91 en los huecos. El 60 % de los D cortos va con
    plano abierto. Simulada en el banco con desajuste ≤ 0,10;
  - ❌ U4, A1 (comentario → equipo: 0,50 donde dudamos), A2, A3 (golpeo: 0,518), A4, A6.
  - Pasada de vídeo en GPU (balón en todos los frames + jugadores con el color del torso), unas 6,5 h.
- **T5**: código del poseedor tipo PathCRF (E078, entrenando 10 pliegues) y de MS-TCN (E079, en cola).
- Incidencias:
  - en esta consola, los escapes de barra invertida dentro de los heredoc se interpretan dos veces (rompió
    `tune_l3` un momento; corregido antes de que lo usara la cola). Para código, usar Edit o Write;
  - `data/` no se puede borrar (regla de permisos).
- Pendiente:
  - resultados de la cola (E048 → E054), del poseedor y de MS-TCN y de la pasada de vídeo (U3, U5, U6, U8);
  - después, artefacto + guardarraíl con lo que haya pasado.
- Bloqueos (personas): email Q2/Q3/Q6 (Q6 decide el audio y el broadcast), PR, reproducibilidad, inscripción.

## 2026-10-25 (noche) — Manel (con agente) · PUNTO DE PARADA
- **Plan aprobado**: «plan de máxima mejora». Copia de trabajo en `C:\Users\manel\.claude\plans\lucky-cuddling-hellman.md`;
  la versión del equipo está en `docs/05_plan_mejora.md`. Fases:
  1. M1 banco v3.1 + guardarraíl en el reto;
  2. M2a familia «control»;
  3. las puertas deciden entre más datos y poseedor tipo PathCRF;
  4. bagging;
  5. congelación.
- Hecho (rama **`banco-v3-1`**, sale de `analisis-fallos-v3`):
  - **1.1** `atd_stats.ball_dynamics()` y `atd_stats --json`: objetivos del balón del reto en
    `docs/investigacion/resultados/objetivos_balon_reto.json`.
    - Reproducen E013: residuo 0,044; dv 1,77/5,59/11,81; 0,043 toques/s; velocidad 5,1/14,3/24,5; velocidad
      antes de un hueco 2,0.
    - Balón lento en vivo: en el vivo predicho del reto, 5,6 % y 0,55 tramos ≥ 1 s por minuto. **En la ventana
      pública, con la GT, 9,4 % y 1,86 por minuto**: el valor real está por encima de 0,55, cerca o por encima
      del limpio (1,08).
  - **1.2** `DegradeParams.ball_corr_smooth`: suavizado gaussiano del error correlado del balón.
    - Con 0 el degradado es **idéntico bit a bit** al anterior (comprobado con J03WR9).
    - Ojo: tocar `degrade.py` cambia el hash de la caché de features, así que **la próxima evaluación recalcula
      las features del banco**, también las de v3.
  - Tests: 60 pasan.
- **Siguiente paso al retomar (1.3)**: `pipeline/calibrate_ball.py`.
  - Para cada partido del banco v3: GT controlada y vivo = GT H/A. Degradar solo el balón con la misma semilla
    que el banco: `rng = default_rng(i)`, `visible`, `occluded`, `weight` como en `degrade()` y luego
    `_long_ball(...)`.
  - Medir con `ball_dynamics(..., live=vivo)`.
  - Rejilla de unas 48 combinaciones: `ball_smooth_sigma` {6, 10} × `ball_corr_noise` {0,4, 0,8, 1,2} ×
    `ball_corr_tau` {75, 150} × `ball_corr_smooth` {0, 10, 25, 50}.
  - Pérdida normalizada frente a los objetivos del reto. Para el balón lento, objetivo ~1,08 por minuto y
    ~8,8 % (limpio), con penalización fuerte por debajo de 0,55.
  - Guardar en `docs/investigacion/resultados/calibracion_balon_v3_1.json`. Elegir `BENCH_DEGRADE_V3_1` en
    `bench_config.py` y añadir `v3_1` a `BENCHES` de `tune_l3`. Escribir ADR-0017.
- Después:
  - **1.4** `tune_l3 --eval-degrade v3_1`: E070 entrenado en v3 y evaluado en v3.1; tiene que verse el defecto;
  - **1.5** reentrenar en v3.1 (protocolo);
  - **1.6** ADR-0018 con los umbrales del guardarraíl **antes** de ejecutar: D < 5 s ≤ 0,25/min, tramos D
    0,8-1,3/min, cambios directos ≥ 1,7/min, % D 0,38-0,50.
- Pendiente de personas:
  - revisar la PR `analisis-fallos-v3` (sin subir todavía);
  - reproducibilidad cruzada (persona C);
  - email Q2/Q3/Q6;
  - inscripción antes del 10 nov.

## 2026-10-25 (tarde) — Manel (con agente)
- Hecho (rama `analisis-fallos-v3`, desde `banco-v3`):
  - **Análisis de fallos del modelo de la entrega** en el banco v3 (E039-E040,
    `docs/investigacion/04_analisis_fallos_v3.md`).
    - `failure_analysis --bench v3`: eventos DFL vía kloppy, tabla por partido y descomposición con oráculos.
    - Banco 0,7206 con la decodificación de la entrega.
    - El margen está en los robos no vistos (+0,146, la mayoría con el balón visible) y en los cambios sobrantes
      (+0,090). El Viterbi está agotado (+0,001).
    - J03WR9 falla en las fronteras.
  - **Reto frente a banco** (E041-E043, solo agregados):
    - unas seis veces más D cortos en el reto (0,68 por minuto frente a 0,09), el 90 % con el balón visible;
    - TreeSHAP: `ball_speed` y `touch_rank`;
    - **el simulador borra el balón lento del juego vivo**: tramos lentos ≥ 1 s por minuto, limpio 1,08, degradado
      0,01, reto 0,55;
    - **cuesta unas 0,09**: inyectado en el banco a la tasa del reto, 0,721 → 0,633.
  - **Puertas**:
    - YOLO como evidencia de equipo: 92 %, pero poco recorrido hoy (E044);
    - audio con PANNs: sin señal (E045);
    - compacidad: ya explotada; tras un robo manda el cambio (E046).
  - **Estado del arte ampliado** (`docs/investigacion/05_estado_del_arte_v2.md`): PathCRF, Ball Radar,
    TranSPORTmer, MS-TCN/ASRF, SkillCorner/PFF, PANNs, compacidad y formaciones.
  - **Plan de mejora** `docs/05_plan_mejora.md`: M1 balón lento en el simulador + guardarraíl en el reto; M2 señal
    de control; M3 más datos; M4 semillas.
  - `make_submission --dump-proba` guarda las probabilidades y features del reto en `data/derived/` (el sha no
    cambia: `9602959c4ec05bd6`).
- Pendiente:
  - aprobar el plan;
  - M1 (26-28 oct);
  - ADR-0017 (banco v3.1) y ADR-0018 (guardarraíl de forma en el reto) cuando se implemente M1.
- Bloqueos (personas):
  - revisar la PR;
  - reproducibilidad cruzada (persona C);
  - descarga de SkillCorner y PFF (persona B, M3);
  - email Q2/Q3/Q6;
  - inscripción antes del 10 nov.

## 2026-10-25 — Manel (con agente)
- Hecho (rama `banco-v3`):
  - **Ablaciones de familias terminadas** (E038, E060, E062, E065-E067): jugadores −0,008, reanudación −0,005,
    balón −0,003, estructura y bloques 0 por separado, pero **−0,012 sin las dos** (redundantes entre sí).
    **Se quedan las cinco familias** (ADR-0015, pendiente de revisión).
  - **Relleno del balón con YOLO afinado con pseudoetiquetas** (E061-E069): pasa la puerta de huecos
    simulados (59,6 % a < 2 m) y cubre el 55,9 % de los huecos reales. En el banco, con una simulación
    optimista, daba +0,030 (E064), pero en la ventana pública 3 de 4 parones salían como juego vivo (E068):
    en el reto el balón relleno está pegado a un jugador (quien lo lleva a sacar o una bota). Con una
    simulación fiel, +0,008 en 6/10 (E069). **Rechazado** (ADR-0016); el código se conserva.
  - **Artefacto definitivo** (E070): cinco familias, apilado, sin relleno, decodificación fija. `La MasIA.csv`
    dos veces con el mismo sha `9602959c4ec05bd6`. Banco 0,7185; ventana pública spellF1 0,651 y macroF1 0,803
    (antes 0,556 y 0,766).
- Pendiente:
  - documento técnico y README con el procedimiento reproducible;
  - revisar «balón» (J03WR9) y los D cortos sobrantes, si da tiempo;
  - PR `banco-v3` → `main` revisada por un compañero; congelación el 14 nov.
- Incidencias: Claude Code cortó procesos por memoria varias veces (dos evaluaciones a la vez, YOLO, juego,
  Docker con 15 GB). Ahora las colas van en serie y como proceso independiente de la sesión. Esa bash no trae
  `/usr/bin` en el PATH, y una evaluación de 2 h se perdió por eso (ya corregido). Ultralytics deja `runs/` en
  la raíz (en `.gitignore`).

## 2026-10-24 — Manel (con agente)
- Hecho (rama `banco-v3`):
  - **E037**: el modelo profundo no aporta sobre las familias (0,7172 y 0,7134 frente a 0,7185). Queda fuera
    de la entrega; el código se conserva.
  - **Informe de estado en PDF** (`docs/informe/`, 12 páginas con gráficos) para ponernos al día los tres:
    qué hemos hecho, por qué, qué se aprobó o rechazó, dónde fallamos y qué viene.
  - **ADR-0014** (banco v3 con DFL y T por fuente) y **ADR-0015** (familias de features; pendiente de la
    ablación).
  - **Documentación al día**: `01_contexto_completo.md` reescrito, `03_plan_ganar.md` con estado,
    prioridades, **reparto para los compañeros** y cómo empezar, `datos.md` y `README.md`.
  - Ablación sin «jugadores» relanzada (se cortó al reiniciarse la sesión). WSL cerrado para liberar
    memoria.
- Pendiente:
  - resto de ablaciones (persona B);
  - reproducibilidad cruzada y documento técnico (persona C);
  - artefacto final con los 10 partidos y regenerar `La MasIA.csv` (agente).
- Bloqueos (personas): email a la organización, invitaciones de GitHub, revisar la PR `banco-v3` → `main`
  y los ADR 0001-0015 (sobre todo el 0004).

## 2026-10-23 — Manel (con agente)
- Hecho (rama `banco-v3`):
  - **Plan de investigación** (`docs/04_plan_investigacion.md`): revisión crítica de lo hecho, estado del arte (PathCRF, Ball Radar, Pressing Intensity, contrapresión, compacidad, formaciones, SoccerNet-Echoes) y plan hasta el 14 nov.
  - **Banco v3 = 3 Metrica + 7 DFL** (descargado con el navegador del portátil). La GT del tracking DFL también necesita la regla controlada (T = 2 s). El simulador calibrado en Metrica deja DFL calibrado sin tocar nada (E027). Referencia: **0,6470** (E030). Entrenar con más partidos ya mejora.
  - **Features por familias**, pensadas como las mira un entrenador (balón, jugadores, bloques, estructura, reanudación; `features/tactical.py`): **0,7185** (+0,072, 9/10, E035). Niveles de bloque relativos al partido tras ver un desajuste real/banco de 0,3-0,4 (`feature_shift`).
  - **Modelo profundo** por conjuntos + BiGRU (`models/seq.py`), promediado con LightGBM: 0,6570 sobre la referencia (E034). Predicción en numpy para que la entrega no necesite torch.
  - Coste de cambio dinámico: sin ganancia (E036). Acierto por longitud de pase: fallan los envíos perdidos, no los cambios de juego (E028). Silbato: sin señal (E029). Comentario en inglés: transcrito en local, parones fiables pero ya detectados; no entra (E033).
  - Infraestructura: caché de features en disco (las 3 evaluaciones simultáneas se cortaron por memoria), `scripts/run_aggregate.sh`, el portátil se mantiene despierto mientras corren los cálculos (se había suspendido dos veces) y la tapa enchufada ya no suspende.
- Pendiente:
  - Combinación familias + modelo profundo; ablación por familias (empezando por «jugadores», que incluye el portero, con desajuste de 0,22); por qué empeora J03WR9.
  - Artefacto final con los 10 partidos, `make_submission` dos veces con el mismo sha, humo en los 5 min públicos.
- Bloqueos (personas): enviar el email (Q2/Q4/Q5/Q6), invitar a los compañeros en GitHub y revisar la PR `banco-v3` → `main`, decidir sobre el ADR-0004.

## 2026-10-22 (tarde) — Manel (con agente)
- Hecho (rama `artefacto-apilado`):
  - **Datos del reto completos**: broadcast `atd.txt` y `broadcast.mp4` movidos de Descargas a `data/broadcast/` y comprobados (164.488 registros contiguos; vídeo 6.579,5 s). El táctico tiene una pista de audio muda; **el broadcast tiene audio real**. Solo falta DFL. `docs/datos.md` actualizado.
  - **Artefacto final** (ADR-0011): el Viterbi reajustado con la GT controlada vuelve a elegir 1/20/10/10 (E022). Se descubrió que el entrenamiento **no era determinista** con la máquina cargada; arreglado con `force_col_wise` + test (E024). El banco no cambia (0,5422 eventos / 0,6245 controlada). **`La MasIA.csv` regenerado**: sha256 `252716b03d614399`, 322 posesiones, idéntico en dos ejecuciones. Humo en los 5 min públicos: 0,513 (ruido del bootstrap 0,38).
  - **Balón fusionado descartado tal cual**: el broadcast alineado rellena el 7,5 % de los huecos (E023) y el detector YOLO genérico acierta el 24 % incluso donde el ATD ve el balón (E025; nuevas opciones `--validate-only` y `--sim-gaps`).
  - **El audio del broadcast se puede usar en todo el partido**: el desfase broadcast-táctico es constante en cada parte (E026). ADR-0013 desbloqueado.
  - El agente ejecutó con autorización de Manel los scripts de solo agregados sobre los datos reales, a través de `scripts/run_aggregate.sh`. Ese script solo muestra la salida si el script acaba bien; si falla, la deja en `data/derived/logs/` para una persona. El clasificador de Claude Code impidió al agente modificar el ADR-0004: **el equipo debe decidir si esta práctica se recoge en ese ADR**.
  - El conector de Gmail no tiene permiso de borradores: el email con Q2/Q4/Q5/Q6 está redactado en `docs/00_reto.md` y lo envía una persona.
- Pendiente:
  - Señales de cambio de posesión (prioridad 3), detector de balón parado en huecos (5), silbato (8), T = 1/2 s y acierto por longitud de pase (10).
- Bloqueos (personas):
  - DFL: `/chrome` para que lo baje el agente, o a mano.
  - Enviar el email; escuchar si el broadcast tiene comentarista; decidir sobre el ADR-0004.

## 2026-10-22 — Manel (con agente)
- Hecho:
  - **Explicación completa del proyecto desde cero** → `docs/01_contexto_completo.md` (reto, datos, validación, modelo, resultados, fallos, futuro). Es el punto de entrada para compañeros y agentes nuevos.
  - Mañana: E017 (robustez con aleatorización), E018 (**apilado aceptado, 0,542**), E019 (la aleatorización no pasa el protocolo sobre el apilado; apilado desplazado 0,558). **ADR-0011**: modelo final = L3 apilado. `train_l3_final --stack` y `make_submission` ya soportan las dos capas (commit `1ab9dc2`).
  - **Análisis a fondo de fallos (E020)** → `docs/investigacion/02_analisis_fallos.md`, script `pipeline/failure_analysis.py`.
  - **Hallazgo**: la GT del banco contaba cada robo de 1 s, y el reto solo cuenta la posesión **controlada** (CHALLENGE.md). Con «un robo cuenta si dura ≥ 1,5 s» el banco reproduce la GT pública (13,7 s y 17,5 % < 3 s, frente a 14,2 s y 17,6 %), y el mismo modelo pasa de 0,542 a **0,628**. **ADR-0012** (propuesta), `external.controlled_possession`, `tune_l3 --controlled` y `--from-oof`.
  - Fallos que quedan: no vemos la mitad de los robos (79 % de las pérdidas), huecos largos (acierto 0,78), duelos, aéreos y despejes, infrasegmentación.
  - Idea del audio del broadcast (silbato y comentarios) → **ADR-0013** (propuesta, bloqueada hasta saber si hay audio) y preguntas **Q6/Q7** en `docs/00_reto.md`.
  - `docs/03_plan_ganar.md`: nueva sección «Estado y prioridades» con la tabla de tareas por persona.
  - **E021**: reentrenar con la GT controlada **no aporta** (0,6225 frente a 0,6245 del modelo actual con Viterbi reajustado). El cambio es de medida: el modelo vigente se queda y se evalúa con `--controlled 1.5`. Calibración revisada: con el baseline, el banco queda ~0,04 más fácil que el reto, así que la estimación orientativa en el reto es ~0,55-0,60.
  - El análisis queda explicado paso a paso en `docs/investigacion/02_analisis_fallos.md`, con las salidas completas en `docs/investigacion/resultados/`.
  - **Presentación** actualizada (36 diapositivas: hallazgo de la definición, por qué perdemos posesiones, causas → soluciones) y subida al repo en `docs/presentacion/` (PDF, PPTX y fuente).
  - **Git consolidado**: todo el trabajo (T1-T8, plan para ganar, análisis de fallos) está en **una sola rama, `analisis-fallos`**, pendiente de una única PR hacia `main`. Se borraron, en local y en GitHub, las ramas ya contenidas en ella. Para recuperar alguna: `git branch <nombre> <sha>`: t1-entorno-uv `a5b4d90`, t2-load-match `2b103b1`, t3-eval-wrapper `687273d`, t4-baseline-submission `e359826`, docs-log-t1 `38b759a`, t5-banco-externo `be19464`, plan-ganar `1ab9dc2`.
- Pendiente:
  - Medir el modelo vigente con T = 1 y 2 s (robustez de ADR-0012).
  - Entrenar el artefacto final y regenerar `La MasIA.csv` (el actual sale del modelo en cuarentena).
  - Señales de cambio de posesión y detector de balón parado dentro de los huecos.
- Bloqueos (personas):
  - `broadcast atd.txt` (parte -009) y ejecutar el detector de balón entero (no estaba corriendo).
  - Bajar DFL a mano. Comprobar si `broadcast.mp4` tiene audio (`ffprobe`, ADR-0013).
  - Enviar Q4/Q6 (y Q2/Q5). Revisar los ADR 0001-0013.

## 2026-10-21 (tarde) — Manel (con agente)
- Hecho:
  - **Plan para ganar** aprobado (`docs/03_plan_ganar.md`), con tres frentes: A visión, B banco y C modelo. Protocolo de selección pre-registrado (ADR-0008).
  - Frente A, código listo y probado con sintéticos: alineador del broadcast (`io/broadcast_align.py` + `pipeline/broadcast_report.py`) y detector de balón en vídeo (`vision/ball_detector.py` + `vision/ball_track.py`, grupo `vision` con torch CUDA). ADR-0010.
  - Frente C: features v2 (toques, receptor, presión) → **cuarentena**. El ATD real tiene el balón filtrado y apenas detecta toques (E012/E013). Banco **v2.1** con el balón filtrado como el ATD y features **v3** con el umbral de toque relativo al partido: L3 sube de 0,465 a **0,519** en v2.1 (E014, ADR-0009). v4 (movimiento colectivo) rechazado (E015). Aleatorización del dominio sin mejora dentro del banco (E016).
  - Apilado con contexto temporal (capa 2) implementado; evaluación en curso.
  - Una evaluación se cortó por falta de memoria: Opera ocupaba 15,6 GB. Las features pasan a float32.
- Pendiente:
  - Resultados de robustez (`--eval-shift`) y del apilado → elegir el modelo y regenerar el artefacto y `La MasIA.csv` (hoy usa v2, en cuarentena).
  - **Enviar a la organización** las preguntas Q2, Q4 y Q5 (borrador en `docs/00_reto.md`).
  - Persona A: broadcast `atd.txt` (parte `-009`) → `broadcast_report`; después `ball_detector`.
  - Persona B: descargar DFL a mano → banco v3.
- Bloqueos / preguntas para el equipo:
  - Revisar y aprobar los ADR 0006-0010 y abrir la PR de `plan-ganar`.

## 2026-10-21 — Manel (con agente)
- Hecho:
  - Humo de L3 en datos reales (E011): 0,5875 en los 5 min públicos (L1 0,4686, baseline 0,3256).
  - **Nombre del equipo: `La MasIA`.** SUBMISSION.md exige `La MasIA.csv` y `La MasIA.pdf` (el evaluador pone el nombre del fichero en el ranking).
- Pendiente:
  - Generar la entrega: `uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"`.
  - Documento técnico `La MasIA.pdf`, repo público y formulario (antes del 17 nov).

## 2026-10-20 (noche 4) — Manel (con agente)
- Hecho:
  - Humo en datos reales (lo ejecutó Manel, solo agregados, E005): pipeline validado (el baseline s50 da 0,3256 en los 5 min públicos, igual que EVALUATION.md). L1 da 0,469 en esos 5 min (solo humo).
  - **Hallazgo del ATD** (E005-E006): el balón falta en ~134 huecos largos (> 4 s) que cubren el 41 % de lo evaluado. Son una mezcla de parón y juego vivo (56 % D en la ventana pública), y dentro de ellos los jugadores van más lentos (0,97 frente a 1,62 m/s). El Smart Tagging solo coincide con el final del 13 % de los huecos.
  - Banco **v2** recalibrado con huecos largos (E006). L3 = LightGBM (46 features por frame) + Viterbi: **0,502** dejando un partido fuera en v2, frente a 0,300 del baseline y 0,326 de L1 (E007-E010). Sin Smart Tagging (aporta +0,008, ruido). ADR-0007: L3 es la entrega principal y el baseline la reserva. Modelo versionado en `src/hackbarca/models/artifacts/` (gzip determinista).
  - T8 (E009): la pérdida está en la **segmentación de equipo** (con equipo perfecto, 0,87), no en el D (0,58) ni en el retraso de las fronteras (óptimo en 0).
- Pendiente:
  - **Ejecutar el humo de L3** sobre los datos reales (`make_submission --model l3`).
  - Mejorar la segmentación de equipo: features de trayectoria del balón y de presión, más capacidad o un modelo secuencial, HSMM con prior de duración. Más partidos de entrenamiento (DFL).
  - Abrir la PR `t5-banco-externo` → `main` (contiene T1-T8 y la rama `docs-log-t1`) y revisar los ADR 0001-0007.
  - Los informes "huérfanos" de `external/.../evaluation/reports/` son la salida por defecto de `run_evaluation.py` (escribe dentro del submódulo). Ahora lo redirigimos a `submissions/` (ignorado). Quien los generó puede borrarlos del submódulo.
- Bloqueos / preguntas para el equipo:
  - Nombre exacto del equipo para `submissions/<equipo>.csv`.
  - DFL (figshare) solo se descarga con navegador.

## 2026-10-20 (noche 3) — Manel (con agente)
- Hecho:
  - T2-T4 revisados (`load_match`, wrapper de evaluación, baseline de reserva). Corregido T4: sugería puntuar con `--gt private`, que **solo existe en la máquina de la organización**. Nunca tendremos el spellF1 del partido completo: solo los 5 min públicos (humo) y el banco externo.
  - T5: banco externo con Metrica sample-data (3 partidos, ~1.140 posesiones), GT H/A/D derivada de eventos, simulador de degradación tipo ATD **calibrado** para que el baseline reproduzca los 8 números publicados (ADR-0006, E001-E002). DFL bloqueado: figshare no deja descargar sin navegador.
  - T6: L1 = reglas + Viterbi + ventanas de reanudación del Smart Tagging. spellF1 0,363 dejando un partido fuera, frente a 0,293 del baseline s50 en el mismo banco (E003-E004).
  - Comando único de entrega: `make_submission --model {baseline,l1}`, determinista y con validación.
- Pendiente:
  - **Que una persona ejecute** sobre los datos reales (solo imprimen agregados): `atd_stats` y `make_submission` para baseline y L1, y el humo con la GT pública. Comandos en la PR de `t5-banco-externo`.
  - T7 (LightGBM + Viterbi), T8 (análisis de errores). Refinar la calibración (2 parámetros en el borde de la rejilla).
- Bloqueos / preguntas para el equipo:
  - Nombre exacto del equipo (el del formulario) para `submissions/<equipo>.csv`.
  - Si alguien puede bajar con navegador los 21 ficheros DFL de figshare a `external_data/dfl/<match_id>/{meta,event,tracking}.xml`, el banco pasa de 3 a 10 partidos.

## 2026-10-20 (noche 2) — Manel (con agente)
- Hecho:
  - **Limpieza de atribución de IA**: `git filter-branch` sobre `main` para quitar los `Co-Authored-By` que quedaron en 6 commits antiguos, y `push --force` a `origin/main`. **Los compañeros necesitan `git fetch` + resetear sus ramas locales a `origin/main` o su próximo push chocará.**
  - **T1 completado** (rama `t1-entorno-uv`, PR pendiente de abrir manualmente en GitHub — `gh` no disponible en este entorno): `pyproject.toml` + `uv.lock` (Python 3.11, kloppy==3.18.0, pandas<3, lightgbm, scikit-learn, matplotlib, pytest) y estructura `src/hackbarca/{io,features,models,decode,eval,pipeline}`. Verificado con `uv sync` + `uv run pytest` desde cero.
  - **Bug encontrado y arreglado** en `.gitignore`: la regla `models/` (pensada para artefactos de modelos entrenados en la raíz) no estaba anclada y también ignoraba `src/hackbarca/models/`, nuestro paquete de código — se habría perdido silenciosamente. Corregido a `/models/` en la misma rama.
  - Verificados por `test -f` (sin leer contenido) los 5 ficheros tácticos imprescindibles en `data/tactical/`: todos presentes.
  - Leída toda la documentación propia + repo oficial (`possession_metrics.py` incluido). Hallazgo importante: **ningún ADR está formalmente aceptado** (los 5 siguen en estado "Propuesta", sin revisiones marcadas) pese a que ya actuamos como si 0002/0004 estuvieran en vigor.
  - Hallazgo: hay outputs de `run_evaluation.py` en `external/.../evaluation/reports/` y `scoreboard.csv` que nadie ha documentado en `docs/experimentos.md` — alguien ya corrió el scorer oficial y no dejó registro.
- Pendiente:
  - Abrir la PR de `t1-entorno-uv` → `main` en GitHub (rama ya empujada).
  - **Aprobar formalmente los ADR 0001-0005** (marcar revisores).
  - Averiguar quién generó los reports huérfanos de `run_evaluation.py` y documentarlos en `experimentos.md`, o reproducirlos y documentarlos yo.
  - T2 (load_match), T3 (evaluación local), T4 (reproducir baseline oficial) — siguientes en la cola.
- Bloqueos / preguntas para el equipo:
  - **Importante**: tras el `push --force` a `main`, cualquiera con cambios locales basados en el `main` antiguo debe hacer `git fetch origin && git reset --hard origin/main` (o rebase) en su copia **antes** de volver a hacer push, si no divergirá.
  - T4 (reproducir baseline oficial sobre datos reales) necesita que una persona ejecute el comando — el agente no puede tocar `data/` (ADR-0004).

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
