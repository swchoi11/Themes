import logging
import time
import functools
from colorlog import ColoredFormatter
from datetime import datetime

APP_LOGGER_NAME = 'THEMES'

def init_logger(
        log_level=logging.INFO,
        log_format=None
):
    """
    콘솔 로거를 초기화합니다.

    Args:
        log_level (int): 로깅 레벨 (기본값: INFO)
        log_format (str): 로그 포맷 문자열 (기본값: None)

    Returns:
        logging.Logger: 초기화된 로거 객체
    """
    if log_format is None:
        log_format = (
            '%(asctime)s - '
            '%(name)s - '
            '%(funcName)s - '
            '%(log_color)s%(levelname)s - '
            '%(message)s'
        )

    formatter = ColoredFormatter(
        log_format,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green', 
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(log_level)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 콘솔 출력 설정
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(log_level)
    logger.addHandler(ch)

    # 파일 출력 설정
    fh = logging.FileHandler(f'log-{datetime.now().strftime("%Y-%m-%d")}.log')
    fh.setFormatter(formatter)
    fh.setLevel(log_level)
    logger.addHandler(fh)


    return logger


def is_initialized(logger_name):
    logger = logging.getLogger(logger_name)
    return len(logger.handlers) > 0


def timefn(fn):
    """
        함수의 실행 시간을 측정하고 로깅하는 데코레이터입니다.
    """

    @functools.wraps(fn)
    def measure_time(*args, **kwargs):
        logger = logging.getLogger(APP_LOGGER_NAME)
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"함수 {fn.__name__} 실행 시간: {execution_time:.2f}초")
        return result

    return measure_time
