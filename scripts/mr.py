#!/usr/bin/env python2.7
'''Poor men's MapReduce.
'''

import argparse
import collections
import glob
import os
import os.path
import pipes
import shutil
import subprocess
import sys
import time

import common

QsubMrOptions = collections.namedtuple('QsubMrOptions', ['name', 'input', 'output', 'numerical_sort', 'retry', 'verbose', 'keep_workdir', 'max_tries', 'mapper', 'mapper_pmem', 'mapper_queue', 'mapper_walltime', 'reducer' ,'reducer_pmem', 'reducer_queue', 'reducer_walltime', 'qsub_args'])

def choose_cat(path):
    if path.endswith('.gz'):
        return 'zcat'
    elif path.endswith('.bz2'):
        return 'bzcat'
    else:
        return 'cat'

bash_start = '''#!/bin/bash
set -e
set -o pipefail
set -o nounset
'''

def format_task_ids(ids):
    parts = []
    low = ids[0]
    high = low
    for i in ids[1:]:
        if i != high + 1:
            parts.append(str(low) if low == high else ('%d-%d' % (low, high)))
            low = i
        high = i
    parts.append(str(low) if low == high else ('%d-%d' % (low, high)))
    return ','.join(parts)

def wait_qsub(qsub_args, poll=15):
    jobid = subprocess.check_output(['qsub'] + qsub_args).strip()
    with open(os.devnull, 'w') as devnull:
        while subprocess.call(['qstat', jobid], stderr=devnull, stdout=devnull) == 0:
            time.sleep(poll)

def basename(path):
    path = os.path.basename(path)
    if path.endswith('.gz'):
        path = path[:-3]
    if path.endswith('.bz2'):
        path = path[:-4]
    if path.endswith('.out'):
        path = path[:-4]
    return path

def run_qsub_mr(options):
    # Record current working directory. Both mapper and reducer commands will run under this directory.
    cwd = os.getcwd()
    # Find input files.
    input_files = glob.glob(options.input)
    common.check(input_files, 'input pattern does not list any file')
    common.check(len(set([basename(i) for i in input_files])) == len(input_files), 'input pattern matches input files with duplicate basename')
    if options.verbose:
        common.info('input pattern %s matched %d files', repr(options.input), len(input_files))
    # Check then create output directory.
    common.check(options.retry or not os.path.exists(options.output), 'output directory already exists')
    workdir = os.path.join(options.output, '.mr')
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    if options.verbose:
        common.info('workdir: %s', workdir)
    # Write out script to rerun this command.
    with common.open(os.path.join(workdir, 'rerun.bash'), 'w') as f:
        f.write('''# Run this script with --rerun is equivalent to running the mapreduction (assuming the input hasn't changed)
        cd %(cwd)s
        %(args)s "$@"
        ''' % {'cwd': pipes.quote(cwd),
               'args': ' '.join(map(pipes.quote, sys.argv))})
    # Write out mapper script to run on nodes.
    with common.open(os.path.join(workdir, 'map.node.bash'), 'w') as f:
        f.write(bash_start + options.mapper)
    # Write out mapper script to submit to qsub.
    with common.open(os.path.join(workdir, 'map.qsub.bash'), 'w') as f:
        f.write(bash_start + '''
        cd %(cwd)s
        if [ -e %(workdir)s/map.${PBS_ARRAYID}.err ]; then
          mv %(workdir)s/map.${PBS_ARRAYID}.err %(workdir)s/map.${PBS_ARRAYID}.err.old
        fi
        bash %(workdir)s/map.${PBS_ARRAYID}.bash 2> %(workdir)s/map.${PBS_ARRAYID}.err
        ''' % {'cwd': pipes.quote(cwd), 'workdir': pipes.quote(workdir)})
    # Write out individual mapper command scripts.
    for i, path in enumerate(input_files):
        with common.open(os.path.join(workdir, 'map.%d.bash' % i), 'w') as f:
            f.write(bash_start + '''
            cd %(cwd)s
            %(cat)s %(path)s | bash %(workdir)s/map.node.bash | gzip - > %(workdir)s/map.%(task_id)d.out.gz
            touch %(workdir)s/map.%(task_id)d.success
            ''' % {'cwd': pipes.quote(cwd),
                   'cat': choose_cat(path),
                   'path': pipes.quote(path),
                   'workdir': pipes.quote(workdir),
                   'task_id': i})
    # Write out reducer script.
    if options.reducer != 'NONE':
        with common.open(os.path.join(workdir, 'reduce.node.bash'), 'w') as f:
            f.write(bash_start + options.reducer)
        with common.open(os.path.join(workdir, 'reduce.qsub.bash'), 'w') as f:
            f.write(bash_start + '''
            cd %(cwd)s
            if [ -e %(workdir)s/reduce.err ]; then
              mv %(workdir)s/reduce.err %(workdir)s/reduce.err.old
            fi
            find %(workdir)s -name 'map.*.out.gz' -print0 2>> %(workdir)s/reduce.err | \\
              xargs -0 zcat 2>> %(workdir)s/reduce.err | \\
              LC_ALL=C sort %(sort_options)s 2>> %(workdir)s/reduce.err | \\
              bash %(workdir)s/reduce.node.bash 2>> %(workdir)s/reduce.err | \\
              gzip - > %(workdir)s/reduce.out.gz 2>> %(workdir)s/reduce.err
            touch %(workdir)s/reduce.success 2>> %(workdir)s/reduce.err
            ''' % {'cwd': pipes.quote(cwd),
                   'workdir': pipes.quote(workdir),
                   'sort_options': '-n' if options.numerical_sort else ''})
    # Run mapper jobs.
    for i in range(options.max_tries):
        # Find tasks to run.
        task_ids = []
        for task_id in range(len(input_files)):
            if not os.path.exists(os.path.join(workdir, 'map.%d.success' % task_id)):
                task_ids.append(task_id)
        if not task_ids:
            break
        qsub_args = ['-N', '%s-map' % options.name, '-q', options.mapper_queue,
                     '-l', 'pmem=%s,walltime=%s' % (options.mapper_pmem, options.mapper_walltime),
                     '-t', format_task_ids(task_ids), '-o', os.devnull, '-e', os.devnull] + \
            options.qsub_args + [os.path.join(workdir, 'map.qsub.bash')]
        if options.verbose:
            common.info('map try %d of %d: need to run %d tasks', i + 1, options.max_tries, len(task_ids))
            common.info('map try %d of %d: qsub_args is %s', i + 1, options.max_tries, repr(qsub_args))
        wait_qsub(qsub_args)
        if options.verbose:
            common.info('map try %d of %d: finished', i + 1, options.max_tries)
    map_success = 0
    for task_id in range(len(input_files)):
        if os.path.exists(os.path.join(workdir, 'map.%d.success' % task_id)):
           map_success += 1
    if i > 0 and options.verbose:
        common.info('map success: %d / %d', map_success, len(input_files))
    common.check(map_success == len(input_files), 'map failed after %d tries', options.max_tries)
    # Run reducer jobs.
    if options.reducer != 'NONE':
        for i in range(options.max_tries):
            if os.path.exists(os.path.join(workdir, 'reduce.success')):
                break
            qsub_args = ['-N', '%s-reduce' % options.name, '-q', options.reducer_queue,
                         '-l', 'pmem=%s,walltime=%s' % (options.reducer_pmem, options.reducer_walltime),
                         '-o', os.devnull, '-e', os.devnull] + options.qsub_args + \
                [os.path.join(workdir, 'reduce.qsub.bash')]
            if options.verbose:
                common.info('reduce try %d of %d: started', i + 1, options.max_tries)
                common.info('reduce try %d of %d: qsub_args is %s', i + 1, options.max_tries, repr(qsub_args))
            wait_qsub(qsub_args)
            if options.verbose:
                common.info('reduce try %d of %d: finished', i + 1, options.max_tries)
        common.check(os.path.exists(os.path.join(workdir, 'reduce.success')), 'reduce failed after %d tries', options.max_tries)
    # Move output.
    if options.reducer != 'NONE':
        src = os.path.join(workdir, 'reduce.out.gz')
        dst = os.path.join(options.output, 'reduce.out.gz')
        subprocess.check_call(['rm', '-f', dst])
        if options.keep_workdir:
            os.symlink(os.path.abspath(src), dst)
        else:
            os.rename(src, dst)
    else:
        for i, path in enumerate(input_files):
            src = os.path.join(workdir, 'map.%d.out.gz' % i)
            dst = os.path.join(options.output, 'map.%s.out.gz' % basename(path))
            subprocess.check_call(['rm', '-f', dst])
            if options.keep_workdir:
                os.symlink(os.path.abspath(src), dst)
            else:
                os.rename(src, dst)
    # Remove workdir
    if not options.keep_workdir:
        shutil.rmtree(workdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', help='name of mapreduction', default='mr')
    parser.add_argument('--input', help='glob pattern for input files', required=True)
    parser.add_argument('--output', help='output directory', required=True)
    parser.add_argument('--numerical_sort', '-n', help="intermediate keys are numeric  (will use 'LC_ALL=C sort -n' instead of 'LC_ALL=C sort')", action='store_true')
    parser.add_argument('--retry', '-r', help='run in "retry mode"; retries jobs that did not succeed according to the *.success files under output directory', action='store_true')
    parser.add_argument('--verbose', '-v', help='verbose output', action='store_true')
    parser.add_argument('--keep_workdir', '-k', help='keep intermediate workdir (.mr under --output) after the job succeeds', action='store_true')
    parser.add_argument('--max_tries', help='maximal number of tries for each single task', type=int, default=5)
    parser.add_argument('--mapper', help='bash script string that works as the mapper', required=True)
    parser.add_argument('--mapper_pmem', help='pmem requirement for running mapper jobs', default='1g')
    parser.add_argument('--mapper_queue', help='job queue for running mapper jobs', default='wide')
    parser.add_argument('--mapper_walltime', help='walltime requirement for running mapper jobs', default='2:00:00')
    parser.add_argument('--reducer', help='bash script string that works as the reducer; or NONE for a map-only job', default='NONE')
    parser.add_argument('--reducer_pmem', help='pmem requirement for running mapper jobs', default='1g')
    parser.add_argument('--reducer_queue', help='job queue for running mapper jobs', default='wide')
    parser.add_argument('--reducer_walltime', help='walltime requirement for running mapper jobs', default='2:00:00')
    parser.add_argument('qsub_args', help='general qsub arguments; overrides --{mapper,reducer}_{pmem,queue,walltime}', nargs='*')
    options = QsubMrOptions(**vars(parser.parse_args()))
    run_qsub_mr(options)
