import sys, logging

_out_file = 'out.log'
_err_file = 'err.log'

log_format = '%(asctime)s %(levelname)s: %(message)s'
date_format = '%x %X'


def setup(gen_ext_logs: bool=False, out_file: str=_out_file, err_file: str=_err_file):

    if gen_ext_logs:
        setup_with_external_logs(out_file, err_file)
    else:
        base_setup()


def base_setup():

    logger = logging.getLogger('cbioge')

    logger.setLevel(logging.DEBUG)

    sout_handler = logging.StreamHandler(sys.stdout)
    sout_handler.setLevel(logging.DEBUG)

    base_format = logging.Formatter(fmt=log_format, datefmt=date_format)
    sout_handler.setFormatter(base_format)

    logger.addHandler(sout_handler)


def setup_with_external_logs(out_file=_out_file, err_file=_err_file):

    base_setup()

    logger = logging.getLogger('cbioge')

    fout_handler = logging.FileHandler(out_file)
    ferr_handler = logging.FileHandler(err_file)

    fout_handler.setLevel(logging.DEBUG)
    ferr_handler.setLevel(logging.ERROR)

    base_format = logging.Formatter(fmt=log_format, datefmt=date_format)
    fout_handler.setFormatter(base_format)
    ferr_handler.setFormatter(base_format)

    fout_filter = LevelFilter(allowed_lvls=[logging.INFO, logging.DEBUG])
    fout_handler.addFilter(fout_filter)

    logger.addHandler(fout_handler)
    logger.addHandler(ferr_handler)


class LevelFilter(logging.Filter):

    def __init__(self, name='', allowed_lvls=[]):
        super().__init__(name)
        self.allowed_lvls = allowed_lvls

    def filter(self, record):
        return record.levelno in self.allowed_lvls