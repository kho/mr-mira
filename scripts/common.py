'''Common python routines.'''

import bz2
import gzip
import logging
import sys

def open(filename, mode='r'):
    '''Opens gzip, bzip2 or plain files automatically based on extension.'''
    if filename.endswith('.gz'):
        ret = gzip.open(filename, mode)
    elif filename.endswith('.bz2'):
        ret = bz2.BZ2File(filename, mode)
    else:
        ret = file(filename, mode)
    return ret

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d: %(message)s')

class FatalException(Exception):
    pass

info = logging.info
warning = logging.warning
error = logging.error
def fatal(msg, *args, **kwargs):
    logging.error(msg, *args, **kwargs)
    raise FatalException

def check(condition, msg, *args, **kwargs):
    if not condition:
        fatal(msg, *args, **kwargs)
