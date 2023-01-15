import logging
from typing import Dict

from . import __name__ as project_name


def setup_logging(project_level: int = logging.DEBUG) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s][%(levelname)8s][%(name)s] %(message)s",
    )
    configs: Dict[str, int] = {
        project_name: project_level,
        "__main__": project_level,
    }
    for logger_name, level in configs.items():
        logging.getLogger(logger_name).setLevel(level)
