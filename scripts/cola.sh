#!/usr/bin/env bash
# Cola de evaluaciones en serie (una a la vez: ~8 GB cada una). Cada linea de external_data/cola.txt:
#   <ID> <argumentos de tune_l3>
# El runner saca la primera linea, la ejecuta con el log en external_data/logs/<ID>.log y apunta el
# resultado en external_data/cola_hecho.txt. Se puede editar cola.txt mientras corre (reordenar,
# anadir o quitar lineas). Si existe external_data/cola.espera, espera a que exista ese fichero de log
# con el resultado final ("real" del time) (p. ej. una evaluacion lanzada fuera de la cola) antes de empezar.
#
#   scripts/cola.sh            # en segundo plano
root="$(cd "$(dirname "$0")/.." && pwd)"
queue="$root/external_data/cola.txt"
done_file="$root/external_data/cola_hecho.txt"
export PATH="/usr/bin:$PATH"
cd "$root" || exit 1
if [ -f "$root/external_data/cola.espera" ]; then
    wait_log="$(cat "$root/external_data/cola.espera")"
    until grep -q "^real\|Traceback" "$wait_log" 2>/dev/null; do sleep 60; done
fi
while true; do
    line="$(grep -v '^\s*$' "$queue" 2>/dev/null | head -1)"
    [ -z "$line" ] && break
    id="${line%% *}"
    args="${line#* }"
    tail -n +2 "$queue" > "$queue.tmp" && mv "$queue.tmp" "$queue"
    start="$(date +%H:%M)"
    (time uv run python -m hackbarca.pipeline.tune_l3 $args) > "$root/external_data/logs/$id.log" 2>&1
    rc=$?
    echo "$id rc=$rc $start-$(date +%H:%M) $args" >> "$done_file"
done
echo "cola vacia $(date +%H:%M)" >> "$done_file"
