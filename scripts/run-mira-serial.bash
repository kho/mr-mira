#!/bin/sh

function usage {
    cat <<EOF
Usage: INPUT=x OUTPUT=y INIT_WEIGHTS=z [ITERS=w METRIC=v MIRA_OPTS=u] $0 ... decoder command ...
EOF
}

CMD=`basename "$0"`
function log {
    echo "$CMD] $@" 1>&2
}

set -e

MAPPER=`which kbest_mirav5`
REDUCER=`which mr_mira_reduce`

[ "x$INPUT" != x ] || { log "Set INPUT"; exit 1; }
[ "x$OUTPUT" != x ] || { log "Set OUTPUT"; exit 1; }
[ "x$INIT_WEIGHTS" != x ] || { log "Set INIT_WEIGHTS"; exit 1; }
log "INPUT = $INPUT"
log "OUTPUT = $OUTPUT"
log "INIT_WEIGHTS = $INIT_WEIGHTS"

if [ "x$ITERS" = x ]; then
    ITERS=50
fi

if [ "x$METRIC" = x ]; then
    METRIC=ibm_bleu
fi

if ! echo "$ITERS" | grep -q ' '; then
    [ ! -e "$OUTPUT" ] || { log "$OUTPUT already exists"; exit 1; }
    mkdir -p "$OUTPUT/0000"
    cp "$INIT_WEIGHTS" "$OUTPUT/0000/weights.out"
fi

for i in `seq $ITERS`; do
    log "ITERATION $i"
    WORKDIR="$OUTPUT/`printf %04d $i`"
    LAST_ITER=$(($i-1))
    LAST_WEIGHTS="$OUTPUT/`printf %04d $LAST_ITER`/weights.out"
    LAST_WEIGHTS=`readlink -f "$LAST_WEIGHTS"`
    mkdir -p "$WORKDIR"
    sort --random-source=/dev/zero -R "$INPUT" | \
	"$MAPPER" -w "$LAST_WEIGHTS" -m "$METRIC" -s 1000 -a -b 1000 -o 2 -p 0 ${MIRA_OPTS} -- "$@" 2> "$WORKDIR/map.log" | \
	LC_ALL=C sort | \
	"$REDUCER" 2> "$WORKDIR/reduce.log" | gzip - > "$WORKDIR/reduce.out.gz"
    n=`zcat "$WORKDIR/reduce.out.gz" | wc -l`
    n=$(($n-1))
    zcat "$WORKDIR/reduce.out.gz" | head -n$n | sort -n | cut -f2 | gzip - > "$WORKDIR/best.out.gz"
    zcat "$WORKDIR/reduce.out.gz" | tail -n1 | sed -e 's/\t/\n/g' | LC_ALL=C sort > "$WORKDIR/weights.out"
done
