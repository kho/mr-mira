#!/bin/bash

# Make sure bash is new enough to support `pipefail`
[ `echo $BASH_VERSION | cut -f1 -d.` -ge 3 ] || { echo "This script requires bash >= 3" 1>&2; exit 1; }

# Exit immediately upon error
set -e

function usage {
    cat <<EOF
Usage: $0 -m mapper -r reducer -i input_dir -o output_dir [-n] [-c] [-p] [-w] [-v] [-k] [-- QSUB_OPTS]

Poor men's MapReduce on qsub.

'mapper' and 'reducer' can be either a single command or a chain of
commands. However, note currently the script always treat the return
value of the last command as the actual return value. 'reducer' may
also be 'NONE', in which case no reduction will be run.

'input_dir' should be a flat directory (with no sub-directories) and
all files inside will be treated as input to the mapper.

Except when in "continue mode", 'output_dir' must either not exist or
be empty.

When in "continue mode", the reduce job is always submitted,
regardless of whether its .success file exists or not.

-n    Intermediate keys are numeric (will use 'LC_ALL=C sort -n' instead of 'LC_ALL=C sort')
-c    "Continue mode"; retries jobs that did not succeed according to the *.success files under 'output_dir'
-p    Mapper is a command that uses pipe (rather than a single command)
-w    Do not exit until all jobs finish
-v    Verbose output
-k    Keep mapper output after the reducer succeeds
EOF
}

function non_empty_dir {
    o=`ls -A "$1"`
    [ -d "$1" ] && [ -n "$o" ]
}

function empty_dir {
    o=`ls -A "$1"`
    [ -d "$1" ] && [ -z "$o" ]
}

function check_opts {
    ([ -d "$input" ] || { echo "Input does not exist or is not a directory: $input" 1>&2; false; }) \
	&& ($continue || [ ! -e "$output" ] || empty_dir "$output" || { echo "Non-empty output already exists: $output" 1>&2; false; }) \
	&& (non_empty_dir "$input" || { echo "Input dir is empty: $input" 1>&2; false; })
}

# Must have qsub
which qsub > /dev/null 2>&1 || { echo "Cannot find qsub!" 1>&2; exit 1; }

mapper="cat"
reducer="cat"
input=`pwd`/in
output=`pwd`/out
sort=sort
continue=false
pipe=false
wait=false
verbose=false
keep=false

while getopts :m:r:i:o:ncpwvkh name; do
    case $name in
	m) mapper=$OPTARG;;
	r) reducer=$OPTARG;;
	i) input=$OPTARG;;
	o) output=$OPTARG;;
	n) sort="sort -n";;
	c) continue=true;;
	p) pipe=true;;
	w) wait=true;;
	v) verbose=true;;
	k) keep=true;;
	h) usage; exit 2;;
	?) echo "$0: unrecognized option -$OPTARG"; exit 2;;
    esac
done

shift $(($OPTIND-1))

if $verbose; then
    echo "Mapper = $mapper" 1>&2
    echo "Reducer = $reducer" 1>&2
    echo "Input dir = $input" 1>&2
    echo "Output dir = $output" 1>&2
    if [ ! -z "$*" ]; then
	echo "qsub options: $@" 1>&2
    fi
    echo "Using $sort for sorting" 1>&2
    if $continue; then
	echo "Continue mode" 1>&2
    fi
    if $pipe; then
	echo "Mapper uses pipe" 1>&2
    fi
    if $wait; then
	echo "Will wait until all jobs finish" 1>&2
    fi
fi

check_opts || exit 1

mkdir -p "$output"

short_output="$(basename $(dirname $(readlink -f $output)))/$(basename $output)"
input=`readlink -f "$input"`
output=`readlink -f "$output"`

alljobs=()

# Submit mapper jobs
mos=()
afterok=""
for i in "$input"/*; do
    bn=`basename "$i"`
    qi=`printf %q "$i"`
    qo=`printf %q "$output/map.$bn.out.gz"`
    qe=`printf %q "$output/map.$bn.err"`
    qs=`printf %q "$output/map.$bn.success"`
    mos+=("$qo")
    # No need to test if we are in continue mode.
    if [ -e "$output/map.$bn.success" ]; then
	continue;
    fi
    cmd=""
    if echo "$i" | grep -q "\.gz$"; then
	cmd="zcat $qi | $mapper"
    elif $pipe; then
	cmd="cat $qi | $mapper"
    else
	cmd="$mapper < $qi"
    fi
    cmd=`cat <<EOF
#!/bin/bash
set -o pipefail
{ $cmd 2> $qe | gzip - > $qo; } && touch $qs
test -e $qs
EOF`
    if $verbose; then
	echo "================================================================================" 1>&2
	echo "$cmd" 1>&2
    fi
    job=`echo "$cmd" | qsub -N "$short_output.map.$bn" -o /dev/null -e /dev/null "$@" | cut -f1 -d.`
    alljobs+=("$job")
    afterok="$afterok:$job"
    sleep 0.1
done

if [ "$reducer" != NONE ]; then
    # Make sure old reduce result no longer exists
    rm -f "$output/reduce.success"
    # Submit the reducer job
    qd=`printf %q "$output"`
    qo=`printf %q "$output/reduce.out.gz"`
    qs=`printf %q "$output/reduce.success"`
    cmd=`cat <<EOF
#!/bin/bash
set -o pipefail
{ zcat ${mos[@]} | LC_ALL=C $sort | $reducer | gzip - > $qo; } && touch $qs
if ! $keep; then find $qd -name 'map.*' -exec rm {} \\; ; fi
test -e $qs
EOF`
    if $verbose; then
	echo "================================================================================" 1>&2
	echo "$cmd" 1>&2
    fi
    job=`echo "$cmd" | qsub -N "$short_output.reduce" -o /dev/null -e "$output/reduce.err" -W depend=afterok$afterok "$@" | cut -f1 -d.`
    alljobs+=("$job")
else
    # There is no reduce job so reduce is immediately successful.
    touch "$output/reduce.success"
fi

if $wait; then
    if $verbose; then
	echo "waiting for ${alljobs[@]} to finish." 1>&2
    fi
    for job in ${alljobs[@]}; do
	while qstat | grep -q "^$job\."; do
	    sleep 15
	done
	if $verbose; then
	    echo "$job finished." 1>&2
	fi
    done
    test -f "$output/reduce.success"
fi
