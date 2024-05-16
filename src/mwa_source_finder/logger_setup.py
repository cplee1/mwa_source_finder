import logging


def get_log_levels() -> dict:
    """Get all available logging levels.

    Returns
    -------
    dict
        A dictionary containing all available logging levels.
    """
    return dict(
        DEBUG=logging.DEBUG,
        INFO=logging.INFO,
        WARNING=logging.WARNING,
        ERROR=logging.ERROR,
        CRITICAL=logging.CRITICAL,
    )


def get_logger(loglvl: int = logging.INFO) -> logging.Logger:
    """Initialise a custom logger.

    Parameters
    ----------
    loglvl : int, optional
        The logging level, by default logging.INFO.

    Returns
    -------
    logging.Logger
        A customised logger object.
    """
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("%(asctime)s %(levelname)-9s :: %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(loglvl)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
