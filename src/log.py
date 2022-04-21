import logging
import os


# a function  to create and save logs in the log files


def logger(path, file):
    """[Create a log file to record the experiment's logs]

    Arguments:
        path {string} -- path to the directory
        file {string} -- file name

    Returns:
        [func] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)
    fname = file.split('.')[0]
    if not os.path.isfile(log_file):
        if os.path.exists(path):
            open(os.path.join(path, file), "w+").close()
        else:
            os.makedirs(path)
            open(os.path.join(path, file), "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger(fname)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers = []
    #logger.propagate = False
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger
