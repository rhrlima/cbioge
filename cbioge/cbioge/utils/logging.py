import sys
import logging


OUT_FILE = 'out.log'
ERR_FILE = 'err.log'

LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
DATE_FORMAT = '%x %X'


def setup(disable_file_logs: bool=False,
    out_file: str=OUT_FILE,
    err_file: str=ERR_FILE
):

    base_logger()
    if not disable_file_logs:
        logger_with_external_files(out_file, err_file)


def base_logger():

    logger = logging.getLogger('cbioge')

    logger.setLevel(logging.DEBUG)

    sout_handler = logging.StreamHandler(sys.stdout)

    base_format = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    sout_handler.setFormatter(base_format)

    logger.addHandler(sout_handler)


def logger_with_external_files(out_file=OUT_FILE, err_file=ERR_FILE):

    logger = logging.getLogger('cbioge')

    fout_handler = logging.FileHandler(out_file)
    ferr_handler = logging.FileHandler(err_file)

    fout_handler.setLevel(logging.DEBUG)
    ferr_handler.setLevel(logging.ERROR)

    base_format = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
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
