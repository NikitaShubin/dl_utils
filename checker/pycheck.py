"""Python-реализация py-конвейера чекера: ruff, mypy, pytest и покрытие."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from checker.common import (
    LEGACY,
    ROOT_FILES,
    TARGETS_MODE,
    Settings,
    Targets,
    collect_targets,
    detect_mode,
    dl_root,
    git_list,
    git_top,
)
from checker.coverage import (
    COVERAGE_JSON,
    Coverage,
    cov_args,
    coverage_available,
)
from checker.coverage import (
    read as read_coverage,
)
from checker.report import GREEN, YELLOW, Reporter, tool_color

# Код возврата при отсутствии файлов подходящего типа (.py, .ipynb):
# возвращается только в режиме тишины (-q); в CI это честный nonzero, чтобы
# покровная проверка не давала ложный «зелёный» по пустому списку файлов:
NO_FILES = 3

USAGE = """Использование: check-py.sh [-f|--fix] [-g|--git-only] [-q] [-H] [путь...]

Позиционные аргументы - проверяемые файлы или папки (.py, .ipynb).
Без путей: запуск из корня dl_utils проверяет белый список,
из любой другой папки - её содержимое.

-f, --fix       разрешить автофиксы (ruff format и ruff check --unsafe-fixes);
                по умолчанию режим отчёта - файлы не изменяются
-g, --git-only  проверять только файлы, закоммиченные в git (удобно для CI)
-q, --quiet-no-files  в режиме тишины (для главного check.sh): при отсутствии
                файлов подходящего типа ничего не печатать и выйти с кодом 3;
                иначе вывести сообщение об отсутствии и выйти с кодом 0
-H, --print-header  печатать шапку (заголовок, версии, цели) - для check.sh;
                без флага шапка подавлена
"""


@dataclass(frozen=True)
class Ctx:
    """Общий контекст прогона: вывод, инструменты, конфиг линтеров."""

    reporter: Reporter
    colors: dict[str, str]
    tools: dict[str, str | None]
    cfg: Path
    fix: bool


def parse_flags(argv: list[str]) -> tuple[Settings, list[str]]:
    """Разбор флагов и целей, как в bash-версии; help/ошибка - SystemExit."""
    fix = git_only = quiet = header = False
    targets: list[str] = []
    for arg in argv:
        if arg in ('-f', '--fix'):
            fix = True
        elif arg in ('-g', '--git-only'):
            git_only = True
        elif arg in ('-q', '--quiet-no-files'):
            quiet = True
        elif arg in ('-H', '--print-header'):
            header = True
        elif arg in ('-h', '--help'):
            sys.stdout.write(USAGE)
            raise SystemExit(0)
        elif arg.startswith('-'):
            sys.stderr.write(f'Неизвестный флаг: {arg}\n')
            sys.stderr.write(USAGE)
            raise SystemExit(2)
        else:
            targets.append(arg)
    return Settings(fix, git_only, quiet, header), targets


def run_process(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Запуск внешнего инструмента с захватом объединённого вывода."""
    return subprocess.run(  # noqa: S603 - исполняемые файлы проверены через which
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )


def _require(ctx: Ctx, name: str) -> str:
    """Абсолютный путь инструмента; без него - ошибка и выход 1."""
    path = ctx.tools.get(name)
    if path is None:
        ctx.reporter.error(f'Инструмент не установлен: {name}')
        raise SystemExit(1)
    return path


def run_linter(ctx: Ctx, name: str, display: str, cmd: list[str], cwd: Path) -> None:
    """Прогон линтера: вывод печатается только при неудаче, как run_linter."""
    tool = Path(cmd[0]).name
    try:
        result = run_process(cmd, cwd)
    except FileNotFoundError:
        ctx.reporter.error(f'Инструмент не установлен: {tool}')
        ctx.reporter.mark_failure(f'{tool}: {display}')
        return
    if result.returncode == 0:
        return
    lines = [
        line
        for line in result.stdout.splitlines()
        if 'may cause conflicts when used with the formatter' not in line
    ]
    if lines:
        sys.stdout.write('\n'.join(lines) + '\n')
    ctx.reporter.mark_failure(f'{name}: {display}')


def check_one_file(
    ctx: Ctx,
    root: Path,
    rel: str,
    coverage: Coverage | None,
    *,
    annotate: bool,
) -> None:
    """Тройка линтеров для одного файла с путями относительно корня цели."""
    display = f'{root}/{rel}'
    suffix = coverage.suffix(root, rel) if annotate and coverage is not None else None
    ctx.reporter.file_line(display, suffix)

    cfg = str(ctx.cfg)
    color = ctx.colors['ruff']
    if ctx.fix:
        format_cmd = [_require(ctx, 'ruff'), 'format', '--config', cfg, color, rel]
    else:
        format_cmd = [
            _require(ctx, 'ruff'),
            'format',
            '--check',
            '--diff',
            '--config',
            cfg,
            color,
            rel,
        ]
    run_linter(ctx, 'ruff format', display, format_cmd, root)

    check_args = ['check', '--config', cfg, color]
    if ctx.fix:
        check_args += ['--fix', '--unsafe-fixes']
    run_linter(
        ctx,
        'ruff check',
        display,
        [_require(ctx, 'ruff'), *check_args, rel],
        root,
    )

    # Mypy: для .ipynb используется обёртка nbqa, т.к. mypy не понимает
    # формат notebook нативно:
    if rel.endswith('.ipynb'):
        mypy_cmd = [_require(ctx, 'nbqa'), 'mypy', rel]
    else:
        mypy_cmd = [
            _require(ctx, 'mypy'),
            '--config-file',
            cfg,
            ctx.colors['mypy'],
            rel,
        ]
    run_linter(ctx, 'mypy', display, mypy_cmd, root)


def run_stage(
    ctx: Ctx,
    label: str,
    bucket: dict[Path, list[str]],
    coverage: Coverage | None,
    *,
    annotate: bool,
) -> None:
    """Прогон тройки линтеров по бакету файлов всех целей, как run_stage."""
    ctx.reporter.separator(label)
    total = sum(len(files) for files in bucket.values())
    if not total:
        ctx.reporter.info('Файлы не найдены')
        return
    failed_files = 0
    for root, files in bucket.items():
        for rel in files:
            before = len(ctx.reporter.failures)
            check_one_file(ctx, root, rel, coverage, annotate=annotate)
            if len(ctx.reporter.failures) > before:
                failed_files += 1
    if failed_files == 0:
        ctx.reporter.success(f'Этап завершён ({total} файлов)')
    else:
        ctx.reporter.error(
            f'Этап завершён с ошибками ({failed_files} из {total} файлов)',
        )


def run_tests(ctx: Ctx, targets: Targets) -> Coverage:
    """Прогон pytest в каждой цели с замером покрытия и сбором метрик."""
    coverage_data = Coverage()
    if not any(targets.test_of.values()):
        ctx.reporter.info('Тесты не найдены - pytest пропущен')
        return coverage_data
    ctx.reporter.separator('Запуск тестов', YELLOW)
    enabled = coverage_available()
    pytest = _require(ctx, 'pytest')
    for root in targets.target_dirs:
        test_files = targets.test_of.get(root, [])
        if not test_files:
            continue
        json_path = root / COVERAGE_JSON
        cmd = [pytest, '-q', '--tb=short', ctx.colors['pytest']]
        if enabled:
            cmd += cov_args(json_path)
        cmd += test_files
        # Вывод pytest захватывается и печатается только при неудаче, чтобы
        # в норме не было шума из имён тестов:
        result = run_process(cmd, root)
        if result.returncode == 0:
            ctx.reporter.success(f'Тесты прошли: {root}')
        else:
            if result.stdout:
                sys.stdout.write(
                    result.stdout + ('\n' if not result.stdout.endswith('\n') else ''),
                )
            ctx.reporter.mark_failure(f'pytest: {root}')
        if enabled and json_path.exists():
            root_coverage = Coverage()
            with suppress(OSError, json.JSONDecodeError):
                root_coverage = read_coverage(root, json_path)
            coverage_data.merge(root_coverage)
            json_path.unlink(missing_ok=True)
            if result.returncode == 0:
                root_line = root_coverage.totals_line()
                if root_line:
                    ctx.reporter.info(f'Покрытие тестами: {root_line}')
    return coverage_data


def _tool_version(bin_path: str | None) -> str:
    """Первая строка --version инструмента или пустая строка при неудаче."""
    if bin_path is None:
        return ''
    result = subprocess.run(  # noqa: S603 - путь к инструменту проверен через which
        [bin_path, '--version'],
        stdout=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        return ''
    return result.stdout.strip()


def print_header(ctx: Ctx, mode: str, display_targets: list[str]) -> None:
    """Шапка прогона: заголовок, версии инструментов, режим и цели."""
    reporter = ctx.reporter
    reporter.box('▶ check-py.sh')
    reporter.step('Запуск проверок качества кода и тестов')

    # Версии инструментов: дрейф версий сразу виден при странных прогонах:
    versions: list[str] = []
    for name in ('ruff', 'mypy', 'nbqa'):
        version = _tool_version(ctx.tools.get(name))
        versions.append(version or f'нет {name}')
    if not coverage_available():
        versions.append('pytest-cov: нет (покрытие не считается)')
    reporter.step('Инструменты: ' + ', '.join(versions))

    if ctx.fix:
        reporter.step('Режим правки (-f): автофиксы разрешены')
    else:
        reporter.step('Режим отчёта: файлы не изменяются (автофиксы - ключ -f)')
    if mode == LEGACY:
        reporter.step('Дефолтный запуск из корня dl_utils - белый список')
    else:
        reporter.step('Целевые пути: ' + ' '.join(display_targets))


def _collect_main(
    settings: Settings,
    targets: list[str],
    tools: dict[str, str | None],
    reporter: Reporter,
) -> Targets:
    """Сбор файлов в бакеты: белый список из корня git или цели по путям."""
    collected = Targets()
    if detect_mode(targets, dl_root()) != LEGACY:
        return collect_targets(targets, tools['git'], settings, reporter)
    # Проверяем Git-репозиторий относительно директории скрипта:
    git = tools['git']
    if git is None or git_top(git, dl_root()) is None:
        reporter.error('Этот скрипт должен запускаться внутри Git-репозитория')
        raise SystemExit(1)
    groot = Path(git_top(git, dl_root()) or '')
    collected.add_root(groot)
    for rel in ROOT_FILES:
        collected.add_file(groot, rel, 'main')
    for rel in git_list(
        git,
        groot,
        ['ls-files', '-c', '-o', '--exclude-standard', '--', 'tests/*.py'],
    ):
        collected.add_file(groot, rel, 'test')
    return collected


def _final_verdict(ctx: Ctx) -> int:
    """Финальный вердикт и код возврата по проваленным этапам."""
    reporter = ctx.reporter
    reporter.separator('ИТОГ')
    if reporter.failures:
        reporter.error(f'Проваленных этапов: {len(reporter.failures)}')
        for stage in reporter.failures:
            reporter.error(f'  - {stage}')
        return 1
    reporter.separator('ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!', GREEN)
    reporter.success('Все проверки завершены успешно!')
    return 0


def main(argv: list[str]) -> int:
    """Основной поток py-конвейера; возвращает код возврата."""
    settings, targets = parse_flags(argv)
    reporter = Reporter()
    colors = {name: tool_color(name) for name in ('ruff', 'mypy', 'pytest')}
    tools = {
        name: shutil.which(name) for name in ('ruff', 'mypy', 'nbqa', 'pytest', 'git')
    }
    ctx = Ctx(reporter, colors, tools, dl_root() / 'pyproject.toml', settings.fix)
    mode = detect_mode(targets, dl_root())
    if mode == TARGETS_MODE and not targets:
        targets = [str(Path.cwd())]
    collected = _collect_main(settings, targets, tools, reporter)

    # Нет файлов подходящего типа: в режиме тишины (-q) ничего не печатаем и
    # отдаём код 3, иначе - инфо-сообщение и код 0:
    if not collected.has_files():
        if settings.quiet:
            return NO_FILES
        reporter.warning('Нет файлов подходящего типа (.py, .ipynb)')
        return 0

    if settings.header:
        print_header(ctx, mode, targets)

    # pytest до линтеров: его прогон даёт метрики покрытия для подписей файлов:
    coverage_data = run_tests(ctx, collected)
    run_stage(
        ctx,
        'Проверка основных файлов',
        collected.main_of,
        coverage_data,
        annotate=True,
    )
    run_stage(ctx, 'Проверка тестов', collected.test_of, coverage_data, annotate=False)
    return _final_verdict(ctx)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
