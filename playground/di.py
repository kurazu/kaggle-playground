from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from injector import Injector, Module

from .printer.di import PrinterModule


def get_injector() -> Injector:
    modules: list[Module] = [PrinterModule()]
    return Injector(modules)


R = TypeVar("R")


def cli_autoinject(func: Callable[..., R]) -> Callable[..., R]:
    @functools.wraps(func)
    def cli_autoinject_wrapper(*args: Any, **kwargs: Any) -> R:
        injector = get_injector()
        return injector.call_with_injection(func, args=args, kwargs=kwargs)

    return cli_autoinject_wrapper
