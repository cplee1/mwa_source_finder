import logging


def get_log_levels():
    return dict(
        DEBUG=logging.DEBUG,
        INFO=logging.INFO,
        WARNING=logging.WARNING,
        ERROR=logging.ERROR,
        CRITICAL=logging.CRITICAL,
    )


def get_logger(loglvl=logging.INFO):
    """Initialise a custom logger.

    Parameters
    ----------
    loglvl : `level`
        The logging level

    Returns
    -------
    logger : `logging.Logger`
        An intance of the logging.Logger class
    """
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(lineno)-4d %(levelname)-9s :: %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(loglvl)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
