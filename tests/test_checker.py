"""Тесты Python-чекера: флаги, сбор целей, отчёт и метрики покрытия."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from checker import coverage as covmod
from checker import report as repmod
from checker.common import (  # noqa: PLC0415
    Settings,
    Targets,
    _collect_git,
    collect_targets,
    detect_mode,
    dl_root,
    git_list,
    git_top,
    is_test_file,
    iter_py_targets,
    lift_tests_root,
)
from checker.pycheck import parse_flags


def test_parse_flags_positions_and_flags() -> None:
    """Позиционные аргументы уходят в цели, флаги - в настройки."""
    settings, targets = parse_flags(['-f', '--git-only', 'path_a', 'path_b'])
    assert settings.fix
    assert settings.git_only
    assert not settings.quiet
    assert not settings.header
    assert targets == ['path_a', 'path_b']


def test_parse_flags_quiet_header() -> None:
    """Флаги тишины и шапки разбираются на месте."""
    settings, targets = parse_flags(['-q', '-H'])
    assert settings.quiet
    assert settings.header
    assert targets == []


def test_parse_flags_clean_cache() -> None:
    """Короткий и длинный флаги очистки кеша разбираются на месте."""
    settings, targets = parse_flags(['-c', '--clean-cache'])
    assert settings.clean_cache
    assert targets == []


def test_parse_flags_unknown_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Неизвестный флаг - сообщение в stderr и код 2."""
    with pytest.raises(SystemExit) as exc_info:
        parse_flags(['--bogus'])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert 'Неизвестный флаг: --bogus' in captured.err
    assert 'Использование' in captured.err


def test_parse_flags_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Флаг -h печатает использование и завершается кодом 0."""
    with pytest.raises(SystemExit) as exc_info:
        parse_flags(['-h'])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert 'Использование' in captured.out


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        (Path('/proj/tests/foo.py'), Path('/proj')),
        (Path('/proj/a/tests/sub/x.py'), Path('/proj/a')),
        (Path('/proj/plain'), Path('/proj/plain')),
        (Path('/tests'), Path('/')),
    ],
)
def test_lift_tests_root(path: Path, expected: Path) -> None:
    """Корень поднимается до родителя ближайшего каталога tests."""
    assert lift_tests_root(path) == expected


def test_is_test_file() -> None:
    """Имена тестов распознаются по префиксу и суффиксу."""
    assert is_test_file('test_utils.py')
    assert is_test_file('labels_test.py')
    assert not is_test_file('utils.py')


def test_read_line_and_branch_metrics(tmp_path: Path) -> None:
    """Метрики покрытия читаются в проценты строк и веток."""
    json_path = tmp_path / 'cov.json'
    file_stats = {
        'summary': {
            'num_statements': 100,
            'covered_lines': 50,
            'num_branches': 10,
            'covered_branches': 5,
        },
    }
    json_path.write_text(
        json.dumps(
            {
                'files': {str(tmp_path / 'a.py'): file_stats},
                'totals': {
                    'num_statements': 100,
                    'covered_lines': 50,
                    'num_branches': 10,
                    'covered_branches': 5,
                },
            },
        ),
        encoding='utf-8',
    )
    cov = covmod.read(tmp_path, json_path)
    assert cov.suffix(tmp_path, 'a.py') == 'строки 50% · ветки 50%'
    assert cov.totals_line() == 'строки 50% · ветки 50%'


def test_physical_path_match_through_symlink(tmp_path: Path) -> None:
    """Метрики находятся по физическому пути при корне-симлинке."""
    real = tmp_path / 'real'
    real.mkdir()
    (real / 'b.py').write_text('x = 1\n', encoding='utf-8')
    link = tmp_path / 'link'
    link.symlink_to(real, target_is_directory=True)
    json_path = tmp_path / 'cov.json'
    file_stats = {
        'summary': {
            'num_statements': 10,
            'covered_lines': 8,
            'num_branches': 0,
        },
    }
    json_path.write_text(
        json.dumps({'files': {str(real / 'b.py'): file_stats}}),
        encoding='utf-8',
    )
    cov = covmod.read(link, json_path)
    assert cov.suffix(link, 'b.py') == 'строки 80%'


def test_suffix_none_without_data(tmp_path: Path) -> None:
    """Без данных покрытия подпись отсутствует."""
    cov = covmod.Coverage()
    assert cov.suffix(tmp_path, 'a.py') is None
    assert cov.totals_line() is None


def test_file_without_statements_skipped(tmp_path: Path) -> None:
    """Файл без исполняемого кода не даёт метрик."""
    json_path = tmp_path / 'cov.json'
    json_path.write_text(
        json.dumps(
            {'files': {str(tmp_path / 'z.py'): {'summary': {'num_statements': 0}}}},
        ),
        encoding='utf-8',
    )
    cov = covmod.read(tmp_path, json_path)
    assert cov.suffix(tmp_path, 'z.py') is None


def test_read_broken_json(tmp_path: Path) -> None:
    """Битый JSON-отчёт вызывает JSONDecodeError."""
    json_path = tmp_path / 'cov.json'
    json_path.write_text('{broken', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        covmod.read(tmp_path, json_path)


def test_cov_args_mount_json_report(tmp_path: Path) -> None:
    """Флаги pytest замеряют и строки, и ветки с JSON-отчётом."""
    json_path = tmp_path / '.coverage_checker.json'
    args = covmod.cov_args(json_path)
    assert '--cov=.' in args
    assert '--cov-branch' in args
    assert f'--cov-report=json:{json_path}' in args


def _git_init(repo: Path) -> None:
    """Инициализация git-репозитория с первым коммитом."""
    git = shutil.which('git')
    assert git is not None
    for args in (
        ['init', '-q'],
        ['add', '-A'],
        ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'init'],
    ):
        subprocess.run([git, '-C', str(repo), *args], check=False)  # noqa: S603


def test_detect_mode_with_targets(tmp_path: Path) -> None:
    """Явные цели всегда дают целевой режим."""
    assert detect_mode(['a.py'], tmp_path) == 'targets'


def test_detect_mode_legacy_from_dl_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Запуск из корня dl_utils без целей - режим белого списка."""
    monkeypatch.setenv('DLUTILS_DIR', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert detect_mode([], dl_root()) == 'legacy'


def test_detect_mode_targets_from_other_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Запуск из чужой папки без целей проверяет её содержимое."""
    monkeypatch.chdir(tmp_path)
    assert detect_mode([], dl_root()) == 'targets'


def test_detect_mode_legacy_oserror_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Права samefile падают на сравнение путей при сломанном cwd."""
    mprint = 'cwd недоступен'

    class LostCwd:
        """Каталог, чей samefile падает по OSError."""

        def __init__(self, value: str) -> None:
            self._value = value

        def __fspath__(self) -> str:
            return self._value

        def samefile(self, _other: object) -> bool:
            raise OSError(mprint)

    monkeypatch.setenv('DLUTILS_DIR', str(tmp_path))
    monkeypatch.setattr(Path, 'cwd', classmethod(lambda _cls: LostCwd(str(tmp_path))))
    assert detect_mode([], dl_root()) == 'legacy'


def test_targets_dedupe_and_sort(tmp_path: Path) -> None:
    """Цели и файлы не дублируются, бакеты сортируются."""
    targets = Targets()
    targets.add_root(tmp_path)
    targets.add_root(tmp_path)
    assert len(targets.target_dirs) == 1
    assert not targets.has_files()
    targets.add_file(tmp_path, 'b.py', 'main')
    targets.add_file(tmp_path, 'b.py', 'main')
    targets.add_file(tmp_path, 'a.py', 'main')
    targets.add_file(tmp_path, 't_test.py', 'test')
    targets.sort_buckets()
    assert targets.main_of[tmp_path] == ['a.py', 'b.py']
    assert targets.test_of[tmp_path] == ['t_test.py']
    assert targets.has_files()


def test_dl_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Корень берётся из DLUTILS_DIR при запуске через обёртку."""
    monkeypatch.setenv('DLUTILS_DIR', str(tmp_path))
    assert dl_root() == tmp_path


def test_dl_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без переменной корень ищется рядом с пакетом."""
    monkeypatch.delenv('DLUTILS_DIR', raising=False)
    assert (dl_root() / 'checker' / '__init__.py').exists()


def test_git_top_none_git(tmp_path: Path) -> None:
    """Без git верх репозитория не определяется."""
    assert git_top(None, tmp_path) is None


def test_git_top_outside_repo(tmp_path: Path) -> None:
    """Вне git-репозитория rev-parse не находит топа."""
    git = shutil.which('git')
    assert git is not None
    assert git_top(git, tmp_path) is None


def test_git_list_and_top_in_repo(tmp_path: Path) -> None:
    """Внутри репозитория находятся топ и закоммиченные файлы."""
    git = shutil.which('git')
    assert git is not None
    (tmp_path / 'a.py').write_text('x = 1\n', encoding='utf-8')
    (tmp_path / 'stray.ipynb').write_text('{}\n', encoding='utf-8')
    _git_init(tmp_path)
    assert git_top(git, tmp_path) == str(tmp_path)
    committed = git_list(git, tmp_path, ['ls-files', '-c', '--', '*.py'])
    assert committed == ['a.py']
    assert git_list(git, tmp_path, ['no-such-command']) == []


def test_iter_py_targets_prunes_and_filters(tmp_path: Path) -> None:
    """Поиск ищет .py/.ipynb и пропускает служебные каталоги."""
    (tmp_path / 'a.py').write_text('', encoding='utf-8')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'b.ipynb').write_text('', encoding='utf-8')
    (tmp_path / '.venv').mkdir()
    (tmp_path / '.venv' / 'hidden.py').write_text('', encoding='utf-8')
    (tmp_path / 'notes.txt').write_text('', encoding='utf-8')
    found = {p.relative_to(tmp_path).as_posix() for p in iter_py_targets(tmp_path)}
    assert found == {'a.py', 'sub/b.ipynb'}


def _settings(*, quiet: bool = False, git_only: bool = False) -> Settings:
    """Настройки: флаги тишины и git-only по умолчанию выключены."""
    return Settings(
        fix=False,
        git_only=git_only,
        quiet=quiet,
        header=False,
        clean_cache=False,
    )


def test_collect_folder_and_lift(tmp_path: Path) -> None:
    """Папка собирается, тесты поднимаются в отдельный бакет."""
    proj = tmp_path / 'proj'
    (proj / 'tests').mkdir(parents=True)
    (proj / 'mod.py').write_text('x = 1\n', encoding='utf-8')
    (proj / 'tests' / 'test_a.py').write_text('assert True\n', encoding='utf-8')
    settings = _settings()
    collected = collect_targets([str(proj)], None, settings, repmod.Reporter())
    assert collected.target_dirs == [proj]
    assert collected.main_of[proj] == ['mod.py']
    assert collected.test_of[proj] == ['tests/test_a.py']


def test_collect_single_file_inside_tests(tmp_path: Path) -> None:
    """Одиночный тестовый файл поднимает корень до родителя tests."""
    proj = tmp_path / 'proj'
    (proj / 'tests').mkdir(parents=True)
    target = proj / 'tests' / 'test_b.py'
    target.write_text('assert True\n', encoding='utf-8')
    collected = collect_targets([str(target)], None, _settings(), repmod.Reporter())
    assert collected.target_dirs == [proj]
    assert collected.test_of[proj] == ['tests/test_b.py']


def test_collect_missing_target(capsys: pytest.CaptureFixture[str]) -> None:
    """Отсутствующий путь - ошибка и выход с кодом 1."""
    with pytest.raises(SystemExit) as exc_info:
        collect_targets(['/no/such/path'], None, _settings(), repmod.Reporter())
    assert exc_info.value.code == 1
    assert 'Путь не найден' in capsys.readouterr().out


def test_collect_non_py_file_ignored(tmp_path: Path) -> None:
    """Файл не по типу не добавляется и не даёт ошибки."""
    target = tmp_path / 'notes.md'
    target.write_text('# note\n', encoding='utf-8')
    settings = _settings()
    collected = collect_targets([str(target)], None, settings, repmod.Reporter())
    assert not collected.has_files()


def test_collect_git_only_uses_committed(tmp_path: Path) -> None:
    """В git-only режиме берутся только закоммиченные файлы."""
    git = shutil.which('git')
    assert git is not None
    (tmp_path / 'mod.py').write_text('x = 1\n', encoding='utf-8')
    _git_init(tmp_path)
    (tmp_path / 'stray.py').write_text('x = 2\n', encoding='utf-8')
    (tmp_path / 'later.py').write_text('x = 3\n', encoding='utf-8')
    collected = collect_targets(
        [str(tmp_path)],
        git,
        _settings(git_only=True),
        repmod.Reporter(),
    )
    assert collected.main_of[tmp_path] == ['mod.py']


def test_collect_git_only_nested_root_trim(tmp_path: Path) -> None:
    """Пути обрезаются до поднятого корня цели в git-only режиме."""
    git = shutil.which('git')
    assert git is not None
    proj = tmp_path / 'proj'
    (proj / 'tests').mkdir(parents=True)
    (proj / 'tests' / 'test_c.py').write_text('assert True\n', encoding='utf-8')
    _git_init(tmp_path)
    collected = collect_targets(
        [str(proj / 'tests')],
        git,
        _settings(git_only=True),
        repmod.Reporter(),
    )
    assert collected.test_of[proj] == ['tests/test_c.py']


def test_collect_git_only_without_git(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """git-only без установленного git - ошибка и выход."""
    settings = _settings(git_only=True)
    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        collect_targets([str(existing)], None, settings, repmod.Reporter())
    assert exc_info.value.code == 1
    assert '--git-only требует git-репозиторий' in capsys.readouterr().out


def test_collect_git_only_outside_repo(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """git-only на не-репозитории - ошибка через неудачный rev-parse."""
    fake_git = tmp_path / 'fake_git'
    fake_git.write_text('#!/usr/bin/env bash\nexit 1\n', encoding='utf-8')
    fake_git.chmod(0o755)
    settings = _settings(git_only=True)
    with pytest.raises(SystemExit) as exc_info:
        collect_targets([str(tmp_path)], str(fake_git), settings, repmod.Reporter())
    assert exc_info.value.code == 1
    assert '--git-only требует git-репозиторий' in capsys.readouterr().out


def test_collect_git_only_single_py_file(tmp_path: Path) -> None:
    """В git-only режиме файл-цель даёт spec только по нему."""
    git = shutil.which('git')
    assert git is not None
    target = tmp_path / 'mod.py'
    target.write_text('x = 1\n', encoding='utf-8')
    _git_init(tmp_path)
    collected = collect_targets(
        [str(target)],
        git,
        _settings(git_only=True),
        repmod.Reporter(),
    )
    assert collected.main_of[tmp_path] == ['mod.py']


def test_collect_git_only_non_py_file_ignored(tmp_path: Path) -> None:
    """В git-only режиме файл-цель не по типу игнорируется."""
    git = shutil.which('git')
    assert git is not None
    target = tmp_path / 'notes.md'
    target.write_text('# note\n', encoding='utf-8')
    _git_init(tmp_path)
    collected = collect_targets(
        [str(target)],
        git,
        _settings(git_only=True),
        repmod.Reporter(),
    )
    assert not collected.main_of.get(tmp_path)
    assert not collected.test_of.get(tmp_path)


def test_collect_git_only_empty_rel_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Пустое rel после обрезки корня подменяется абсолютным путём."""
    proj = tmp_path / 'proj'
    proj.mkdir()
    monkeypatch.setattr('checker.common.git_top', lambda *_: str(tmp_path))
    monkeypatch.setattr('checker.common.git_list', lambda *_: ['proj/'])
    collected = Targets()
    collected.add_root(proj)
    _collect_git(collected, proj, proj, tmp_path, 'git', repmod.Reporter())
    assert collected.main_of[proj] == [str(proj)]


def test_collect_disk_warns_outside_git(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Вне репозитория предупреждаем о выборе всех файлов с диска."""
    (tmp_path / 'mod.py').write_text('x = 1\n', encoding='utf-8')
    collected = collect_targets([str(tmp_path)], None, _settings(), repmod.Reporter())
    assert collected.has_files()
    assert 'Вне git-репозитория' in capsys.readouterr().out


def test_collect_disk_quiet_suppresses_warning(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """В тихом режиме предупреждение вне репозитория подавлено."""
    (tmp_path / 'mod.py').write_text('x = 1\n', encoding='utf-8')
    collected = collect_targets(
        [str(tmp_path)],
        None,
        _settings(quiet=True),
        repmod.Reporter(),
    )
    assert collected.has_files()
    assert 'Вне git-репозитория' not in capsys.readouterr().out


def test_totals_line_lines_only() -> None:
    """Без веток итоговая строка содержит только строки."""
    cov = covmod.Coverage()
    cov.totals.statements = 10
    cov.totals.covered_lines = 5
    assert cov.totals_line() == 'строки 50%'


def test_coverage_merge_totals() -> None:
    """Счётчики и метрики соседних корней суммируются."""
    first = covmod.Coverage()
    second = covmod.Coverage()
    first.totals.statements = 10
    first.totals.covered_lines = 5
    second.totals.statements = 10
    second.totals.branches = 4
    second.totals.covered_branches = 2
    second.metrics['a.py'] = covmod.FileMetrics(50, None)
    first.merge(second)
    assert first.metrics['a.py'] == covmod.FileMetrics(50, None)
    assert first.totals_line() == 'строки 25% · ветки 50%'


def test_file_metrics_display_branches() -> None:
    """Подпись входит и с ветками, и без них."""
    assert covmod.FileMetrics(45, None).display() == 'строки 45%'
    assert covmod.FileMetrics(45, 60).display() == 'строки 45% · ветки 60%'


def test_metrics_bad_types_guarded(tmp_path: Path) -> None:
    """Неверные типы в отчёте не роняют чтение метрик."""
    json_path = tmp_path / 'cov.json'
    stats = {
        'summary': {
            'num_statements': 'nope',
            'covered_lines': 'nope',
            'num_branches': 'nope',
            'covered_branches': 'nope',
        },
    }
    json_path.write_text(
        json.dumps(
            {
                'files': {str(tmp_path / 'a.py'): stats},
                'totals': {'num_statements': 'x', 'covered_lines': 'y'},
            },
        ),
        encoding='utf-8',
    )
    cov = covmod.read(tmp_path, json_path)
    assert cov.suffix(tmp_path, 'a.py') is None
    assert cov.totals_line() is None


def test_suffix_broken_symlink(tmp_path: Path) -> None:
    """Битая ссылка не даёт метрик и не роняет подпись."""
    (tmp_path / 'broken').symlink_to('/no/such/target/file')
    cov = covmod.Coverage()
    assert cov.suffix(tmp_path, 'broken') is None


def test_column_signature_physical_symlink(tmp_path: Path) -> None:
    """Подпись находится по физическому пути через симлинк на файл."""
    (tmp_path / 'real.py').write_text('x = 1\n', encoding='utf-8')
    (tmp_path / 'alias.py').symlink_to(tmp_path / 'real.py')
    cov = covmod.Coverage()
    cov.metrics['real.py'] = covmod.FileMetrics(50, None)
    assert cov.suffix(tmp_path, 'alias.py') == 'строки 50%'


def test_read_metrics_edge_cases(tmp_path: Path) -> None:
    """Некорректные секции отчёта не роняют чтение и игнорируются."""
    json_path = tmp_path / 'cov.json'
    json_path.write_text(
        json.dumps(
            {
                'files': {
                    str(tmp_path / 'a.py'): ['not-a-dict'],
                    str(tmp_path / 'b.py'): {'summary': [1, 2]},
                    str(tmp_path / 'c.py'): {
                        'summary': {
                            'num_statements': 10,
                            'covered_lines': 'nope',
                            'num_branches': 4,
                            'covered_branches': 'nope',
                        },
                    },
                },
                'totals': ['not-a-dict'],
            },
        ),
        encoding='utf-8',
    )
    cov = covmod.read(tmp_path, json_path)
    assert cov.suffix(tmp_path, 'a.py') is None
    assert cov.suffix(tmp_path, 'b.py') is None
    assert cov.suffix(tmp_path, 'c.py') == 'строки 0% · ветки 0%'
    assert cov.totals_line() is None


def test_read_skips_physical_path_outside_root(tmp_path: Path) -> None:
    """Физический путь вне корня не добавляется в ключи метрик."""
    outside = tmp_path.parent / 'outside.py'
    outside.write_text('x = 1\n', encoding='utf-8')
    (tmp_path / 'ref.py').symlink_to(outside)
    json_path = tmp_path / 'cov.json'
    file_stats = {'summary': {'num_statements': 10, 'covered_lines': 5}}
    json_path.write_text(
        json.dumps({'files': {str(tmp_path / 'ref.py'): file_stats}}),
        encoding='utf-8',
    )
    cov = covmod.read(tmp_path, json_path)
    assert cov.suffix(tmp_path, 'ref.py') == 'строки 50%'


def test_coverage_available_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без pytest-cov покрытие распознаётся как недоступное."""
    monkeypatch.setattr(covmod.importlib.util, 'find_spec', lambda _name: None)
    assert not covmod.coverage_available()


def test_reporter_messages(capsys: pytest.CaptureFixture[str]) -> None:
    """Ярлыки сообщений отчёта присутствуют в выводе."""
    reporter = repmod.Reporter()
    reporter.info('инфо')
    reporter.success('успех')
    reporter.warning('предупреждение')
    reporter.error('ошибка')
    reporter.step('шаг')
    out = capsys.readouterr().out
    assert 'INFO' in out
    assert 'SUCCESS' in out
    assert 'WARNING' in out
    assert 'ERROR' in out
    assert '🔹' in out


def test_reporter_file_line(capsys: pytest.CaptureFixture[str]) -> None:
    """Строка файла несёт подпись метрик, когда они есть."""
    reporter = repmod.Reporter()
    reporter.file_line('/proj/mod.py')
    reporter.file_line('/proj/mod.py', 'строки 50%')
    out = capsys.readouterr().out
    assert '/proj/mod.py' in out
    assert 'строки 50%' in out


def test_reporter_separator_box_long_text(capsys: pytest.CaptureFixture[str]) -> None:
    """Линии и рамка строятся даже для очень длинного текста."""
    reporter = repmod.Reporter()
    reporter.separator('ИТОГ')
    reporter.separator('📏' * 90)
    reporter.box('▶ check-py')
    out = capsys.readouterr().out
    assert 'ИТОГ' in out
    assert '▶ check-py' in out


def test_reporter_mark_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """Провал записывается в реестр и печатается."""
    reporter = repmod.Reporter()
    reporter.mark_failure('mypy: /proj/mod.py')
    assert reporter.failures == ['mypy: /proj/mod.py']
    assert 'Этап провален' in capsys.readouterr().out


def test_tool_color_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без терминала инструменты получают опцию выключения цвета."""

    class Term:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, 'stderr', Term())
    assert repmod.tool_color('ruff') == '--color=never'
    assert repmod.tool_color('mypy') == '--no-color-output'
    assert repmod.tool_color('pytest') == '--color=no'
    assert repmod.tool_color('unknown') == ''


def test_tool_color_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """С терминалом цвет включается принудительно."""

    class Term:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, 'stderr', Term())
    assert repmod.tool_color('ruff') == '--color=always'
    assert repmod.tool_color('mypy') == '--color-output'
    assert repmod.tool_color('pytest') == '--color=yes'
