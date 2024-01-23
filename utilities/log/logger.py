import configparser
import datetime
import inspect
import logging
import os
import sys
from pathlib import Path

from decorators.singleton import Singleton
from utilities import file_utils
from utilities.log.color import RESET, BOLD, YELLOW, RED
from utilities.log.logparam import LogParams

logger = logging.getLogger(__name__)

import time


def setup_logger(args):
    _logger = Logger.instance()

    if hasattr(args, 'conf_file') and not args.conf_file is None:
        _logger.configure(args.conf_file)

    if hasattr(args, 'output_dir'):
        date_str = datetime.datetime.now().strftime("%Y-%b-%d__%H-%M-%S")
        log_file_path = os.path.join(args.output_dir, f'{date_str}.log')

    logging_level = args.logging_level if hasattr(args, 'logging_level') else 'debug'
    _logger.set_logging_config(logging_level, log_file_path)
    return _logger


def current_milli_time():
    return int(round(time.time() * 1000))


def __adjust_file_handler_logging_level(file_handler, logging_level):
    logging_level = logging_level.lower()

    if LogParams.NO_LOGS == logging_level:
        Logger.is_enabled = False
        return

    if logging_level == LogParams.DEBUG:
        file_handler.setLevel(logging.DEBUG)
    elif logging_level == LogParams.INFO:
        file_handler.setLevel(logging.INFO)
    elif logging_level == LogParams.ERROR:
        file_handler.setLevel(logging.ERROR)


@Singleton
class Logger(object):
    start_time = current_milli_time()
    logging_level = "debug"
    is_enabled = True
    log_file_path = None

    def configure(self, config_file_path=None):
        config = configparser.ConfigParser()
        if not os.path.exists(config_file_path): return
        config.read(config_file_path)

        try:
            self.logging_level = config.get('log', 'severity_level')
            self.is_enabled = config.getboolean('log', 'is_enabled')
            self.set_logging_config(self.logging_level)
        except Exception as e:
            print(f'could not read configuration file: {e}')
            sys.exit(1)
        pass

    def set_logging_config(self, logging_level, log_file_path):
        self.__set_loging_level(logging_level)
        self.log_file_path = log_file_path

        Logger.is_initialized_stream_handler = True
        streamHandler = logging.StreamHandler(sys.stdout)
        logger.addHandler(streamHandler)

        file_utils.generate_directory_if_not_exists(Path(self.log_file_path).parent)
        file_handler = logging.FileHandler(self.log_file_path)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def debug(self, msg, color=RESET):
        if self.is_enabled:
            sys.stdout.write(color)
            logger.debug(self.__assemble_log_entry(msg))
            sys.stdout.write(RESET)

    def info(self, msg, color=BOLD):
        if self.is_enabled:
            sys.stdout.write(color)
            logger.info(self.__assemble_log_entry(msg))
            sys.stdout.write(RESET)

    def warning(self, msg):
        if self.is_enabled:
            sys.stdout.write(YELLOW)
            logger.warning(self.__assemble_log_entry(msg))
            sys.stdout.write(RESET)

    def error(self, msg):
        if self.is_enabled:
            sys.stdout.write(RED)
            logger.error(self.__assemble_log_entry(msg))
            sys.stdout.write(RESET)

    @staticmethod
    def __assemble_log_entry(msg):
        from inspect import getframeinfo
        frameinfo = getframeinfo(inspect.stack()[2][0])
        pos_index = frameinfo.filename.rfind(os.sep) + 1
        file_name = frameinfo.filename[pos_index:]
        str = '{}, line {}: {}'.format(file_name, frameinfo.lineno, msg)
        return str

    def __set_loging_level(self, logging_level):
        self.logging_level = logging_level
        logging_level = logging_level.lower()

        if LogParams.NO_LOGS == logging_level:
            Logger.is_enabled = False
            return

        if logging_level == LogParams.DEBUG:
            logger.setLevel(logging.DEBUG)
        elif logging_level == LogParams.INFO:
            logger.setLevel(logging.INFO)
        elif logging_level == LogParams.ERROR:
            logger.setLevel(logging.ERROR)
