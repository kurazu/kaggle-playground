import logging

logger = logging.getLogger(__name__)


class LoggingPrinter:
    def print(self, message: str) -> None:
        logger.info(message)
