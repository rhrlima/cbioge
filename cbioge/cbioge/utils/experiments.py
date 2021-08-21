import os, argparse, platform

from . import checkpoint as ckpt
from . import logging as cbio_logging


MAX_GPU_MEMORY = 0.8


def _limit_gpu_memory(fraction=MAX_GPU_MEMORY):
    '''Limits GPU memory to use tensorflow-gpu'''
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=tfconfig))


def check_os(): # TODO remove
    # Checks OS to limit GPU memory and avoid errors
    if platform.system() == 'Windows':
        _limit_gpu_memory()


def basic_setup(default_folder=ckpt.ckpt_folder, external_log=False):
    '''Uses argparse to add some basic flags when running from command line

    This basic setup adds:
    *   -c/--checkpoint : defines the checkpoints folder
    *   -o/--output : name the output.log file
    *   -e/--error : name the error.log file
    *   -l/--log : flag if log generates files or not
    '''

    args = argparse.ArgumentParser(prog='simple_args')
    args.add_argument('-c', '--checkpoint', type=str, default=default_folder)
    args.add_argument('-o', '--output', type=str, default='out.log')
    args.add_argument('-e', '--error', type=str, default='err.log')
    args.add_argument('-l', '--log', type=bool, default=external_log)

    parser = args.parse_args()

    ckpt.ckpt_folder = parser.checkpoint

    if not os.path.exists(ckpt.ckpt_folder):
        os.makedirs(ckpt.ckpt_folder)

    cbio_logging.setup(external_log, 
        out_file=os.path.join(ckpt.ckpt_folder, parser.output), 
        err_file=os.path.join(ckpt.ckpt_folder, parser.error))