"""Общие данные и сбор целей для py-конвейера чекера."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from checker import report

LEGACY: Literal['legacy'] = 'legacy'
TARGETS_MODE: Literal['targets'] = 'targets'

# Файлы из корня репозитория, которые проверяются в дефолтном режиме запуска
# без аргументов из корня dl_utils; временный костыль - со временем от белых
# списков планируется отказаться совсем:
ROOT_FILES: list[str] = [
    'labels.py',
    'pt_utils.py',
    'onnx_utils.py',
    'ollm_utils.py',
    'boxmot_utils.py',
    'ul_utils.py',
    'sam3al.py',
    'docker/set_symbolic_flag.py',
    'PreAnnotation/SAM3.ipynb',
    'PreAnnotation/Detection.ipynb',
    'PreAnnotation/Segmentation.ipynb',
    'checker/__init__.py',
    'checker/report.py',
    'checker/common.py',
    'checker/coverage.py',
    'checker/pycheck.py',
]

# Каталоги, исключаемые из поиска файлов на диске:
PRUNE_NAMES = frozenset(
    {
        '.git',
        '__pycache__',
        '.venv',
        'venv',
        'node_modules',
        'dist',
        'build',
        '.mypy_cache',
        '.pytest_cache',
        '.ruff_cache',
        # Транзитные артефакты Jupyter - не подлежат проверке:
        '.ipynb_checkpoints',
    },
)

# Типы файлов, за которые отвечает py-конвейер:
PY_EXTS = ('.py', '.ipynb')


@dataclass(frozen=True)
class Settings:
    """Разобранные флаги командной строки."""

    fix: bool
    git_only: bool
    quiet: bool
    header: bool
    clean_cache: bool


class Targets:
    """Собранные по целям файлы: корни и бакеты «основные» и «тесты»."""

    def __init__(self) -> None:
        """Пустые корни и бакеты."""
        self.target_dirs: list[Path] = []
        self.main_of: dict[Path, list[str]] = {}
        self.test_of: dict[Path, list[str]] = {}
        self._seen_files: set[tuple[str, str]] = set()

    def add_root(self, root: Path) -> None:
        """Регистрация корня цели без дублей."""
        if root not in self.main_of:
            self.target_dirs.append(root)
            self.main_of[root] = []
            self.test_of[root] = []

    def add_file(self, root: Path, rel: str, kind: str) -> None:
        """Добавление файла в бакет цели без дублей."""
        key = (str(root), rel)
        if key in self._seen_files:
            return
        self._seen_files.add(key)
        (self.test_of if kind == 'test' else self.main_of)[root].append(rel)

    def has_files(self) -> bool:
        """Есть ли в целях хоть один файл подходящего типа."""
        return any(
            self.main_of.get(root) or self.test_of.get(root)
            for root in self.target_dirs
        )

    def sort_buckets(self) -> None:
        """Детерминированный порядок файлов внутри каждой цели."""
        for root in self.target_dirs:
            self.main_of[root] = sorted(set(self.main_of[root]))
            self.test_of[root] = sorted(set(self.test_of[root]))


def dl_root() -> Path:
    """Корень dl_utils: из переменной окружения обёртки или рядом с пакетом."""
    env = os.environ.get('DLUTILS_DIR')
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent


def detect_mode(targets: list[str], script_root: Path) -> Literal['legacy', 'targets']:
    """Режим работы: белый список из корня dl_utils или цели по путям."""
    if targets:
        return TARGETS_MODE
    try:
        same = Path.cwd().samefile(script_root)
    except OSError:
        same = os.path.normpath(Path.cwd()) == os.path.normpath(script_root)
    return LEGACY if same else TARGETS_MODE


def lift_tests_root(path: Path) -> Path:
    """Подъём корня до родителя каталога tests, как lift_tests_root."""
    d = path
    while d.parent != d and d.name != 'tests':
        d = d.parent
    if d.parent != d:
        return d.parent
    return path


def is_test_file(name: str) -> bool:
    """Относится ли имя файла к тестам по соглашению имён."""
    return name.startswith('test_') or name.endswith('_test.py')


def git_top(git: str | None, cwd: Path) -> str | None:
    """Верхний каталог git-репозитория или None вне репозитория."""
    if git is None:
        return None
    r = subprocess.run(  # noqa: S603 - путь к git проверен через which
        [git, '-C', str(cwd), 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    if r.returncode:
        return None
    return r.stdout.strip()


def git_list(git: str, cwd: Path, args: list[str]) -> list[str]:
    """Список путей от git ls-files; при неудаче - пусто, как `|| true`."""
    r = subprocess.run(  # noqa: S603 - путь к git проверен через which
        [git, '-C', str(cwd), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    if r.returncode:
        return []
    return r.stdout.splitlines()


def iter_py_targets(root: Path) -> list[Path]:
    """Файлы .py/.ipynb под каталогом без служебных подкаталогов."""
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in PRUNE_NAMES)
        matches += [
            Path(dirpath) / name for name in filenames if Path(name).suffix in PY_EXTS
        ]
    return matches


def collect_targets(
    targets: list[str],
    git: str | None,
    settings: Settings,
    report_: report.Reporter,
) -> Targets:
    """Сбор файлов целей в структуру «корень -> относительные пути»."""
    collected = Targets()
    for item in targets:
        _collect_one(collected, Path(item), git, settings, report_)
    collected.sort_buckets()
    return collected


def _collect_one(
    collected: Targets,
    target: Path,
    git: str | None,
    settings: Settings,
    report_: report.Reporter,
) -> None:
    """Обработка одной цели: определение корня и сбор файлов."""
    if not target.exists():
        report_.error(f'Путь не найден: {target}')
        sys.exit(1)
    abs_path = target.resolve()
    root = lift_tests_root(abs_path if abs_path.is_dir() else abs_path.parent)
    collected.add_root(root)
    git_ctx = abs_path if abs_path.is_dir() else abs_path.parent
    if settings.git_only:
        _collect_git(collected, abs_path, root, git_ctx, git, report_)
    else:
        _collect_disk(
            collected,
            abs_path,
            root,
            git_ctx,
            git,
            quiet=settings.quiet,
            report_=report_,
        )


def _collect_git(  # noqa: PLR0913, PLR0917 - параметры корня и git как в bash
    collected: Targets,
    abs_path: Path,
    root: Path,
    git_ctx: Path,
    git: str | None,
    report_: report.Reporter,
) -> None:
    """Сбор только закоммиченных файлов через git ls-files."""
    if git is None:
        report_.error(f'--git-only требует git-репозиторий, а цель вне его: {abs_path}')
        sys.exit(1)
    groot = git_top(git, git_ctx)
    if groot is None:
        report_.error(f'--git-only требует git-репозиторий, а цель вне его: {abs_path}')
        sys.exit(1)
    gtop = Path(groot)
    prefix = os.path.relpath(abs_path, gtop)
    root_grel = os.path.relpath(root, gtop)
    if root_grel == os.curdir:
        root_grel = ''
    if abs_path.is_file():
        if abs_path.suffix not in PY_EXTS:
            return
        spec = [prefix]
    else:
        spec = [f'{prefix}/*.py', f'{prefix}/*.ipynb']
    # git отдаёт пути от корня репозитория - обрезаем префикс поднятого корня
    # цели, чтобы получить путь относительно него:
    for p in git_list(git, gtop, ['ls-files', '-c', '--', *spec]):
        rel = p
        if root_grel and p.startswith(f'{root_grel}/'):
            rel = p[len(root_grel) + 1 :]
        if not rel:
            rel = str(abs_path).removeprefix(f'{root}/')
        _add(collected, root, rel)


def _collect_disk(  # noqa: PLR0913, PLR0917 - параметры корня и git как в bash
    collected: Targets,
    abs_path: Path,
    root: Path,
    git_ctx: Path,
    git: str | None,
    *,
    quiet: bool,
    report_: report.Reporter,
) -> None:
    """Сбор всех найденных на диске файлов за вычетом служебных каталогов."""
    if not quiet and (git is None or git_top(git, git_ctx) is None):
        report_.warning(f'Вне git-репозитория - берутся все файлы с диска: {abs_path}')
    if abs_path.is_file():
        if abs_path.suffix in PY_EXTS:
            _add(collected, root, os.path.relpath(abs_path, root))
        return
    for p in iter_py_targets(abs_path):
        _add(collected, root, os.path.relpath(p, root))


def _add(collected: Targets, root: Path, rel: str) -> None:
    """Классификация файла по бакету «основные» или «тесты»."""
    name = rel.rsplit('/', 1)[-1]
    kind = 'test' if is_test_file(name) else 'main'
    collected.add_file(root, rel, kind)
