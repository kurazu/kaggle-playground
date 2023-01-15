import logging

import click
from injector import Inject

from ..di import cli_autoinject
from ..logs import setup_logging
from ..printer.interface import PrinterInterface

logger = logging.getLogger(__name__)


@click.command()
@click.option("--message", type=str, required=True, default="Hello, world!")
@cli_autoinject
def main(printer: Inject[PrinterInterface], message: str) -> None:
    printer.print(message)


if __name__ == "__main__":
    setup_logging()
    main()
