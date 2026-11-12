#!/usr/bin/env bash
# Ejecuta un script sobre los datos del reto y muestra su salida SOLO si acaba bien (ADR-0004).
#
#   scripts/run_aggregate.sh <nombre> <comando...>
#   scripts/run_aggregate.sh make_submission uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"
#
# Los scripts permitidos imprimen solo agregados. Si uno falla, su traza puede contener lineas de
# datos: queda en data/derived/logs/<nombre>.log (bloqueado a los agentes) y la lee una persona.
name="$1"; shift
root="$(cd "$(dirname "$0")/.." && pwd)"
log="$root/data/derived/logs/$name.log"
mkdir -p "$(dirname "$log")"
cd "$root" && "$@" > "$log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
    grep -v -i "warning" "$log"
else
    echo "FALLO exit=$rc -> lo revisa una persona: data/derived/logs/$name.log"
fi
exit $rc
