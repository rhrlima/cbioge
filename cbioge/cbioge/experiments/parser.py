import os
import argparse

from ..utils import logging
from ..utils import checkpoint as ckpt

base_parser = argparse.ArgumentParser()

DEFAULTS = {
    'checkpoint': 'checkpoints',

    'grammar': None,
    'dataset': None,

    'train-size': None,
    'valid-size': None,
    'test-size': None,
    'valid-split': None,

    'epochs': 10,
    'batch': 32,

    'pop': 10,
    'evals': 20,

    'selection': None,
    't-size': 2,

    'crossover': None,
    'cross-rate': 0.8,

    'mutation': None,
    'mut-rate': 0.1,

    'replace': None,
    'elites': 0.25,

    'custom-op': None,
    'op-rate': 0.6,
}


def basic_args(defaults={}, avoid_parse=False):
    '''Uses argparse to add some basic flags when running from command line

    The default setup has:
    *   -c  --checkpoint - defines the checkpoints folder
    *   -l  --log - flag if log generates files or not
    *   -o  --output - name the output.log file
    *   -e  --error - name the error.log file
    *   -v  --verbose - flag that includes DEBUG messages to logs'''

    _overwrite_defaults(defaults)

    base_parser.add_argument('-c', '--checkpoint',
        type=str,
        default=DEFAULTS['checkpoint'])

    base_parser.add_argument('-l', '--no-logs',
        action='store_true')

    base_parser.add_argument('-o', '--output',
        type=str,
        default='out.log')

    base_parser.add_argument('-e', '--error',
        type=str,
        default='err.log')

    base_parser.add_argument('-v', '--verbose',
        action='store_true')

    if not avoid_parse:
        parse_default_args()


def parse_default_args():
    args = base_parser.parse_args()

    ckpt.CKPT_FOLDER = args.checkpoint

    if not os.path.exists(ckpt.CKPT_FOLDER):
        os.makedirs(ckpt.CKPT_FOLDER)

    logging.setup(args.no_logs,
        out_file=os.path.join(ckpt.CKPT_FOLDER, args.output),
        err_file=os.path.join(ckpt.CKPT_FOLDER, args.error))


def _overwrite_defaults(new_args):
    for key, value in new_args.items():
        # if key in DEFAULTS:
        # replaces or add new key/value
        DEFAULTS[key] = value


def evolution_args(defaults={}):

    _overwrite_defaults(defaults)

    # configures the default args global to all experiments
    basic_args(defaults, True)

    # grammar args
    base_parser.add_argument('--grammar', type=str, default=DEFAULTS['grammar'])

    # dataset args
    base_parser.add_argument('--dataset', type=str, default=DEFAULTS['dataset'])
    base_parser.add_argument('--train-size', type=int, default=DEFAULTS['train-size'])
    base_parser.add_argument('--valid-size', type=int, default=DEFAULTS['valid-size'])
    base_parser.add_argument('--test-size', type=int, default=DEFAULTS['test-size'])
    base_parser.add_argument('--valid-split', type=float, default=DEFAULTS['valid-split'])

    # problem args
    base_parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    base_parser.add_argument('--batch', type=int, default=DEFAULTS['batch'])

    # search args
    base_parser.add_argument('--pop', type=int, default=DEFAULTS['pop'])
    base_parser.add_argument('--evals', type=int, default=DEFAULTS['evals'])

    base_parser.add_argument('--selection',
        type=str,
        default=DEFAULTS['selection'],
        choices=['tournament'])

    base_parser.add_argument('--t-size', type=int, default=DEFAULTS['t-size'])

    base_parser.add_argument('--crossover',
        type=str,
        default=DEFAULTS['crossover'],
        choices=['onepoint', 'gene', 'none'])

    base_parser.add_argument('--cross-rate', type=float, default=DEFAULTS['cross-rate'])

    base_parser.add_argument('--mutation',
        type=str,
        default=DEFAULTS['mutation'],
        choices=['point', 'term', 'nonterm', 'none'])

    base_parser.add_argument('--mut-rate', type=float, default=DEFAULTS['mut-rate'])

    base_parser.add_argument('--replace',
        type=str,
        default=DEFAULTS['replace'],
        choices=['worst', 'elitsm'])

    base_parser.add_argument('--elites', type=float, default=DEFAULTS['elites'])

    base_parser.add_argument('--custom-op',
        type=str,
        default=DEFAULTS['custom-op'],
        choices=['halfhalf', 'halfchoice'])

    base_parser.add_argument('--op-rate', type=float, default=DEFAULTS['op-rate'])

    parse_default_args()
    return base_parser.parse_args()
