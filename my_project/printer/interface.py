from typing import Protocol


class PrinterInterface(Protocol):
    def print(self, message: str) -> None:
        ...
