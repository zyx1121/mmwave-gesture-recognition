import sys

from .console.cli import Console
from .utils.logger import logger


def main() -> None:
    try:
        Console().cmdloop()

    except KeyboardInterrupt:
        sys.exit(0)

    except Exception as e:
        logger.error(f"[red]{e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
