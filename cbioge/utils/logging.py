import sys, logging

_out_file = 'out.log'#os.path.join(ckpt.ckpt_folder, 'out.log')
_err_file = 'err.log'#os.path.join(ckpt.ckpt_folder, 'err.log')

log_format = '%(asctime)s %(levelname)s: %(message)s'
date_format = '%x %X'


def setup(out_file=_out_file, err_file=_err_file, log_lvl=logging.INFO):

    logger = logging.getLogger('cbioge')

    logger.setLevel(log_lvl)

    sout_handler = logging.StreamHandler(sys.stdout)
    #serr_handler = logging.StreamHandler(sys.stderr)
    fout_handler = logging.FileHandler(out_file)
    ferr_handler = logging.FileHandler(err_file)

    sout_handler.setLevel(logging.DEBUG)
    fout_handler.setLevel(logging.INFO)
    ferr_handler.setLevel(logging.ERROR)

    base_format = logging.Formatter(fmt=log_format, datefmt=date_format)
    sout_handler.setFormatter(base_format)
    fout_handler.setFormatter(base_format)
    ferr_handler.setFormatter(base_format)

    base_filter = LevelFilter(allowed_lvls=[logging.INFO, logging.DEBUG])
    sout_handler.addFilter(base_filter)
    fout_handler.addFilter(base_filter)

    logger.addHandler(sout_handler)
    logger.addHandler(fout_handler)
    logger.addHandler(ferr_handler)


class LevelFilter(logging.Filter):

    def __init__(self, name='', allowed_lvls=[]):
        super().__init__(name)
        self.allowed_lvls = allowed_lvls

    def filter(self, record):
        return record.levelno in self.allowed_lvls