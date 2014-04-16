#!/bin/sh
# FIXME: much of the code duplicates with run-mira-qsub.bash. Find a way to make things more modularized!

function usage {
    cat <<EOF
Usage: INPUT=x OUTPUT=y [INIT_WEIGHTS=z ITERS=50 METRIC=ibm_bleu J=10 MIRA_OPTS=''] $0 ... decoder command ...
Use ITERS="x y" to restart from previous runs, otherwise INIT_WEIGHTS is required.
EOF
    exit 1
}

function log {
    echo "$(date)] $@" 1>&2
}

set -e

MAPPER=`which kbest_mirav5`
REDUCER=`which mr_mira_reduce`

[ "x$INPUT" != x ] || { log "Set INPUT"; usage; }
[ "x$OUTPUT" != x ] || { log "Set OUTPUT"; usage; }
log "INPUT = $INPUT"
log "OUTPUT = $OUTPUT"

[ "x$ITERS" != x ] || ITERS=50
[ "x$METRIC" != x ] || METRIC=ibm_bleu
[ "x$J" != x ] || J=10
if ! echo "$ITERS" | grep -q ' '; then
    [ "x$INIT_WEIGHTS" != x ] || { log "Set INIT_WEIGHTS when ITERS=$ITERS!"; usage; }
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
    r=0
    mkdir -p "$WORKDIR"
    for j in "$INPUT"/*; do
	if echo "$j" | grep -q "\.gz$"; then
	    cmd="zcat $(printf %q "$j") |"
	else
	    cmd="cat $(printf %q "$j") |"
	fi
	name="$(basename "$j")"
	cmd="$cmd $MAPPER -w $LAST_WEIGHTS -m $METRIC -s 1000 -a -b 1000 -o 2 -p 0 $MIRA_OPTS -- $* && touch $(printf %q "$WORKDIR/map.$name.success")"
	echo "$cmd" | sh 2> "$WORKDIR/map.$(basename "$j").err" | gzip - > "$WORKDIR/map.$(basename "$j").out.gz" &
	r=$(($r+1))
	if [ "$r" -eq "$J" ]; then
	    wait
	    r=0
	fi
    done
    wait
    for j in "$INPUT"/*; do
	name="$(basename "$j")"
	if [ ! -e "$WORKDIR/map.$name.success" ]; then
	    log "map.$name did not succeed"
	    exit 1
	fi
    done
    zcat "$WORKDIR"/map.*.out.gz | LC_ALL=C sort | "$REDUCER" | gzip - > "$WORKDIR/reduce.out.gz"
    n=`zcat "$WORKDIR/reduce.out.gz" | wc -l`
    n=$(($n-1))
    zcat "$WORKDIR/reduce.out.gz" | head -n$n | sort -n | cut -f2 | gzip - > "$WORKDIR/best.out.gz"
    zcat "$WORKDIR/reduce.out.gz" | tail -n1 | sed -e 's/\t/\n/g' | LC_ALL=C sort > "$WORKDIR/weights.out"
    rm "$WORKDIR/map."*
done
