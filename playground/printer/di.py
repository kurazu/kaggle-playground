from injector import Module, provider

from .impl import LoggingPrinter
from .interface import PrinterInterface


class PrinterModule(Module):
    @provider
    def provide_printer(self, printer: LoggingPrinter) -> PrinterInterface:
        return printer
