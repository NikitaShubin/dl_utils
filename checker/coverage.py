"""Измерение покрытия тестами и поперфайловые метрики для отчёта чекера."""

from __future__ import annotations

import importlib.util
import json
import os
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

# Имя временного JSON-отчёта coverage в корне цели; удаляется после разбора:
COVERAGE_JSON = '.coverage_checker.json'


def coverage_available() -> bool:
    """Доступен ли pytest-cov для замера покрытия."""
    return importlib.util.find_spec('pytest_cov') is not None


def cov_args(json_path: Path) -> list[str]:
    """Флаги pytest для замера строк и веток с JSON-отчётом."""
    return ['--cov=.', '--cov-branch', f'--cov-report=json:{json_path}']


@dataclass(frozen=True)
class FileMetrics:
    """Проценты покрытия одного файла - строки и, возможно, ветки."""

    lines_pct: int
    branches_pct: int | None

    def display(self) -> str:
        """Подпись для строки файла, как «строки 45% · ветки 60%»."""
        if self.branches_pct is None:
            return f'строки {self.lines_pct}%'
        return f'строки {self.lines_pct}% · ветки {self.branches_pct}%'


@dataclass
class Totals:
    """Суммарные счётчики покрытия по всем корням целей."""

    statements: int = 0
    covered_lines: int = 0
    branches: int = 0
    covered_branches: int = 0

    def add(self, other: Totals) -> None:
        """Агрегация счётчиков другого источника."""
        self.statements += other.statements
        self.covered_lines += other.covered_lines
        self.branches += other.branches
        self.covered_branches += other.covered_branches


@dataclass
class Coverage:
    """Метрики покрытия: карта по относительным путям и итоговые счётчики."""

    metrics: dict[str, FileMetrics] = field(default_factory=dict)
    totals: Totals = field(default_factory=Totals)

    def merge(self, other: Coverage) -> None:
        """Слияние метрик и счётчиков соседнего корня."""
        self.metrics.update(other.metrics)
        self.totals.add(other.totals)

    def suffix(self, root: Path, rel: str) -> str | None:
        """Подпись метрик файла по логическому и физическому пути, или None."""
        found = self.metrics.get(os.path.normpath(rel))
        if found is not None:
            return found.display()
        with suppress(OSError):
            physical = os.path.normpath(
                os.path.relpath((root / rel).resolve(), root.resolve()),
            )
            if not physical.startswith(os.pardir):
                found = self.metrics.get(physical)
                if found is not None:
                    return found.display()
        return None

    def totals_line(self) -> str | None:
        """Итоговая строка «строки N% · ветки M%» или None без данных."""
        parts: list[str] = []
        if self.totals.statements:
            lines = round(self.totals.covered_lines * 100 / self.totals.statements)
            parts.append(f'строки {lines}%')
        if self.totals.branches:
            branches = round(self.totals.covered_branches * 100 / self.totals.branches)
            parts.append(f'ветки {branches}%')
        if not parts:
            return None
        return ' · '.join(parts)


def read(root: Path, json_path: Path) -> Coverage:
    """Разбор JSON-отчёта coverage в метрики с двойным ключом путей."""
    data = json.loads(json_path.read_text(encoding='utf-8'))
    cov = Coverage()
    for file_key, file_stats in data.get('files', {}).items():
        metrics = _metrics_from(file_stats)
        if metrics is None:
            continue
        for key in _path_keys(root, Path(file_key)):
            cov.metrics[key] = metrics
    totals = data.get('totals', {})
    if isinstance(totals, dict):
        cov.totals = _totals_from(totals)
    return cov


def _metrics_from(stats: object) -> FileMetrics | None:
    """Метрики файла из его секции отчёта или None без исполняемого кода."""
    if not isinstance(stats, dict):
        return None
    summary = stats.get('summary')
    if not isinstance(summary, dict):
        return None
    num = summary.get('num_statements', 0)
    if not isinstance(num, int) or num <= 0:
        return None
    covered = summary.get('covered_lines', 0)
    if not isinstance(covered, int):
        covered = 0
    num_branches = summary.get('num_branches', 0)
    branches_pct = None
    if isinstance(num_branches, int) and num_branches > 0:
        covered_branches = summary.get('covered_branches', 0)
        if not isinstance(covered_branches, int):
            covered_branches = 0
        branches_pct = round(covered_branches * 100 / num_branches)
    return FileMetrics(round(covered * 100 / num), branches_pct)


def _path_keys(root: Path, file_path: Path) -> list[str]:
    """Ключи метрик файла: логический и физический относительный путь."""
    keys: list[str] = []
    logical = os.path.normpath(os.path.relpath(file_path, root))
    if not logical.startswith(os.pardir):
        keys.append(logical)
    with suppress(OSError):
        physical = os.path.normpath(
            os.path.relpath(file_path.resolve(), root.resolve()),
        )
        if not physical.startswith(os.pardir):
            keys.append(physical)
    return keys


def _totals_from(totals: dict[str, object]) -> Totals:
    """Итоговые счётчики покрытия из секции totals отчёта."""
    return Totals(
        statements=_int(totals, 'num_statements'),
        covered_lines=_int(totals, 'covered_lines'),
        branches=_int(totals, 'num_branches'),
        covered_branches=_int(totals, 'covered_branches'),
    )


def _int(totals: dict[str, object], key: str) -> int:
    """Целочисленное значение счётчика с защитой от битых данных."""
    value = totals.get(key, 0)
    return value if isinstance(value, int) else 0
