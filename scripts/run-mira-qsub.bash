#!/bin/sh

function usage {
    cat <<EOF
Usage: INPUT=x OUTPUT=y [INIT_WEIGHTS=z ITERS=50 METRIC=ibm_bleu MEM=4g MAXTRY=5 MIRA_OPTS=''] $0 ... decoder command ...
Use ITERS="x y" to restart from previous runs, otherwise INIT_WEIGHTS is required.
EOF
}

function log {
    echo "$(date)] $@" 1>&2
}

set -e

SUBMIT=`which qsub-mr.bash`
MAPPER=`which kbest_mirav5`
REDUCER=`which mr_mira_reduce`

[ "x$INPUT" != x ] || { log "Set INPUT"; usage; exit 1; }
[ "x$OUTPUT" != x ] || { log "Set OUTPUT"; usage; exit 1; }
[ "x$MEM" != x ] || MEM=4g
[ "x$MAXTRY" != x ] || MAXTRY=5
[ "x$ITERS" != x ] || ITERS=50
[ "x$METRIC" != x ] || METRIC=ibm_bleu
if ! echo "$ITERS" | grep -q ' '; then
    [ "x$INIT_WEIGHTS" != x ] || { log "Set INIT_WEIGHTS"; usage; exit 1; }
fi

log "INPUT = $INPUT"
log "OUTPUT = $OUTPUT"
log "INIT_WEIGHTS = $INIT_WEIGHTS"

QSUB_OPTS=(-q wide -l pmem="$MEM",walltime=2:00:00)

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
    if [ ! -e "$LAST_WEIGHTS" ]; then
	log "cannot find $LAST_WEIGHTS"
	exit 1
    fi
    for j in $(seq "$MAXTRY"); do
	if "$SUBMIT" \
	    -m "$MAPPER -w $LAST_WEIGHTS -m $METRIC -s 1000 -a -b 1000 -o 2 -p 0 $MIRA_OPTS -- $*" \
	    -r "$REDUCER" \
	    -i "$INPUT" \
	    -o "$WORKDIR" \
	    -w -c -- "${QSUB_OPTS[@]}"; then
	    break
	fi
	echo "Try $j failed"
    done
    if [ ! -e "$WORKDIR/reduce.success" ]; then
	log "all $MAXTRY runs failed"
	exit 1
    fi

    n=`zcat "$WORKDIR/reduce.out.gz" | wc -l`
    n=$(($n-1))
    gzip "$WORKDIR"/*.err &
    zcat "$WORKDIR/reduce.out.gz" | head -n$n | sort -n | cut -f2 | gzip - > "$WORKDIR/best.out.gz" &
    zcat "$WORKDIR/reduce.out.gz" | tail -n1 | sed -e 's/\t/\n/g' | LC_ALL=C sort > "$WORKDIR/weights.out"
done
