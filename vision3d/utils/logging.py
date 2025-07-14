import logging

def configure_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger with the specified name and logging level.
    Use custom format for log messages.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (default is INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger