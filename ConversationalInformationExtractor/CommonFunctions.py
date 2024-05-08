import logging
from time import time_ns


def time_wrapper(local_func):
    local_handler = create_custom_logger(time_wrapper.__name__)

    def local_wrapper(*args, **kwargs):
        start_time = time_ns()
        results = local_func(*args, **kwargs)
        end_time = time_ns()
        local_handler.debug(f"Time taken for {local_func.__name__} execution: "
                           f"{(end_time - start_time) / (1000 * 1000):.3f} msec")
        return results
    return local_wrapper


def create_custom_logger(fn_name):
    local_logger = logging.getLogger(fn_name)
    local_logger.setLevel(logging.INFO)
    if not local_logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        local_formatter = logging.Formatter("%(asctime)s --> %(filename)s -> %(funcName)s -> %(levelname)s %(message)s",
                                            datefmt="%d/%m/%y %I:%M:%S %p")
        console_handler.setFormatter(local_formatter)
        local_logger.addHandler(console_handler)
    return local_logger
