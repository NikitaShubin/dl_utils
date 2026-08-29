"""Вывод py-конвейера чекера: цвета, блоки отчёта, реестр проваленных этапов."""

from __future__ import annotations

import shutil
import sys

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
GRAY = '\033[0;90m'
MAGENTA = '\033[0;35m'
NC = '\033[0m'


def _width() -> int:
    """Ширина терминала в колонках, как `tput cols` с запасом 80."""
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def tool_color(name: str) -> str:
    """Флаг окраски для внешнего инструмента: авто-детект по терминалу stderr."""
    flags = {
        'ruff': ('--color=always', '--color=never'),
        'mypy': ('--color-output', '--no-color-output'),
        'pytest': ('--color=yes', '--color=no'),
    }
    pair = flags.get(name)
    if pair is None:
        return ''
    return pair[0] if sys.stderr.isatty() else pair[1]


def _spaces(width: int) -> str:
    """Отступ пробелами неотрицательной ширины для текста в рамке."""
    return ' ' * max(0, width)


class Reporter:
    """Вывод сообщений и реестр проваленных этапов, как в utils.sh."""

    def __init__(self) -> None:
        """Пустой реестр проваленных этапов."""
        self.failures: list[str] = []

    def separator(self, text: str, color: str = CYAN) -> None:
        """Сплошная линия ═ с текстом по центру, как print_separator."""
        total = _spaces(_width() - 3 - len(text))
        side = len(total) // 2
        left = '═' * side
        right = '═' * (len(total) - side)
        sys.stdout.write(f'\n{color}{left} {text} {right}{NC}\n\n')

    def box(self, text: str, color: str = CYAN) -> None:
        """Прямоугольник из псевдографики с текстом по центру, как print_box."""
        inner = _width() - 3
        lpad = (inner - len(text)) // 2
        rpad = inner - len(text) - lpad
        hline = '═' * max(0, inner)
        empty = _spaces(inner)
        sys.stdout.write(
            f'\n{color}╔{hline}╗{NC}\n'
            f'{color}║{empty}║{NC}\n'
            f'{color}║{_spaces(lpad)}{text}{_spaces(rpad)}║{NC}\n'
            f'{color}║{empty}║{NC}\n'
            f'{color}╚{hline}╝{NC}\n\n',
        )

    def info(self, text: str) -> None:
        """Информационная строка с ярлыком INFO."""
        sys.stdout.write(f'{CYAN}ℹ️  INFO:{NC} {text}\n')

    def success(self, text: str) -> None:
        """Строка успеха с ярлыком SUCCESS."""
        sys.stdout.write(f'{GREEN}✅ SUCCESS:{NC} {text}\n')

    def warning(self, text: str) -> None:
        """Строка предупреждения с ярлыком WARNING."""
        sys.stdout.write(f'{YELLOW}⚠️  WARNING:{NC} {text}\n')

    def error(self, text: str) -> None:
        """Строка ошибки с ярлыком ERROR."""
        sys.stdout.write(f'{RED}❌ ERROR:{NC} {text}\n')

    def step(self, text: str) -> None:
        """Строка шага из шапки."""
        sys.stdout.write(f'{CYAN}🔹 {text}{NC}\n')

    def file_line(self, display: str, suffix: str | None = None) -> None:
        """Строка проверяемого файла с необязательной подписью метрик."""
        line = f'{CYAN}▸ {MAGENTA}{display}{NC}'
        if suffix:
            line += f'  {GRAY}({suffix}){NC}'
        sys.stdout.write(line + '\n')

    def mark_failure(self, stage: str) -> None:
        """Фиксация проваленного этапа с сообщением, как mark_failure."""
        self.failures.append(stage)
        self.error(f'Этап провален: {stage}')
