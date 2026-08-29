"""Тесты конвейера: линтеры, pytest, шапка и основной поток на фейковых инструментах."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from checker import coverage as covmod
from checker.common import LEGACY, ROOT_FILES, Targets
from checker.pycheck import (
    NO_FILES,
    Ctx,
    check_one_file,
    main,
    print_header,
    run_linter,
    run_process,
    run_stage,
    run_tests,
)
from checker.report import Reporter, tool_color

if TYPE_CHECKING:
    from pathlib import Path


def _script(bins: Path, name: str, body: str) -> None:
    """Создание исполняемого bash-скрипта с шебангом."""
    path = bins / name
    path.write_text(body, encoding='utf-8')
    path.chmod(0o755)


def _make_bins(tmp_path: Path, proj: Path, *, with_json: bool = True) -> Path:
    """Фейковые инструменты, управляемые маркером FAIL в проверяемых файлах."""
    bins = tmp_path / 'bin'
    bins.mkdir()
    _script(
        bins,
        'ruff',
        """#!/usr/bin/env bash
set -u
[[ "${1:-}" == --version ]] && { echo "ruff fake 1.0"; exit 0; }
last="${@: -1}"
[[ -f "$last" ]] && grep -qs FAIL "$last" && { echo "ruf fail: $last"; exit 1; }
exit 0
""",
    )
    _script(
        bins,
        'mypy',
        """#!/usr/bin/env bash
set -u
[[ "${1:-}" == --version ]] && { echo "mypy fake 1.0"; exit 0; }
last="${@: -1}"
[[ -f "$last" ]] && grep -qs FAIL "$last" && { echo "mypy fail: $last"; exit 1; }
exit 0
""",
    )
    _script(bins, 'nbqa', f'#!/usr/bin/env bash\nshift\nexec {bins}/mypy "$@"\n')
    if with_json:
        doc = json.dumps(
            {
                'files': {
                    str(proj / 'src' / 'mod.py'): {
                        'summary': {
                            'num_statements': 8,
                            'covered_lines': 8,
                            'num_branches': 0,
                        },
                    },
                },
                'totals': {
                    'num_statements': 8,
                    'covered_lines': 8,
                    'num_branches': 0,
                    'covered_branches': 0,
                },
            },
        )
        _script(
            bins,
            'pytest',
            f"""#!/usr/bin/env bash
set -u
out=""
fail=""
for a in "$@"; do
  case "$a" in
    --cov-report=json:*) out="${{a#--cov-report=json:}}" ;;
    *) [[ -f "$a" ]] && grep -qs FAIL "$a" && fail=1 ;;
  esac
done
echo "pytest-output"
[[ -n "$out" ]] && printf '%s' '{doc}' > "$out"
[[ -z "$fail" ]] && exit 0 || exit 1
""",
        )
    else:
        _script(bins, 'pytest', '#!/usr/bin/env bash\necho "pytest-output"\nexit 0\n')
    return bins


def _ctx(
    bins: Path,
    cfg: Path,
    *,
    fix: bool = False,
    tools: dict[str, str | None] | None = None,
) -> Ctx:
    """Контекст конвейера на наборе фейковых инструментов."""
    present: dict[str, str | None] = {
        name: str(bins / name) for name in ('ruff', 'mypy', 'nbqa', 'pytest', 'git')
    }
    if tools:
        present.update(tools)
    colors = {name: tool_color(name) for name in ('ruff', 'mypy', 'pytest')}
    return Ctx(Reporter(), colors, present, cfg, fix)


def _make_proj(tmp_path: Path, *, failing: bool = False) -> Path:
    """Мини-проект в git: src/mod.py и тесты для него."""
    proj = tmp_path / 'proj'
    (proj / 'src').mkdir(parents=True)
    marker = 'FAIL' if failing else 'ok'
    (proj / 'src' / 'mod.py').write_text(
        '"""Модуль фикстуры."""\n\n\ndef add(a: int, b: int) -> int:\n'
        f'    """Складывает числа: {marker}."""\n    return a + b\n',
        encoding='utf-8',
    )
    (proj / 'tests').mkdir()
    (proj / 'tests' / 'test_mod.py').write_text(
        '"""Тесты модуля."""\n\n\ndef test_add() -> None:\n    assert True\n',
        encoding='utf-8',
    )
    (proj / 'pyproject.toml').write_text(
        '[project]\nname = "fixture"\nversion = "0.1"\n',
        encoding='utf-8',
    )
    _git(proj)
    return proj


def _make_failing_proj(tmp_path: Path) -> Path:
    """Провальный мини-проект со вторым именем корня."""
    proj = tmp_path / 'proj_fail'
    (proj / 'src').mkdir(parents=True)
    (proj / 'src' / 'mod.py').write_text(
        '"""Провальный модуль."""\n\n\nX = FAIL\n',
        encoding='utf-8',
    )
    return proj


def _git(repo: Path) -> None:
    """Инициализация и первый коммит репозитория фикстуры."""
    git = shutil.which('git')
    assert git is not None
    for args in (
        ['init', '-q'],
        ['add', '-A'],
        ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'init'],
    ):
        subprocess.run([git, '-C', str(repo), *args], check=False)  # noqa: S603


def _patch_path(monkeypatch: pytest.MonkeyPatch, bins: Path) -> None:
    """Префикс PATH каталогом фейковых инструментов."""
    monkeypatch.setenv('PATH', f'{bins}{os.pathsep}{os.environ.get("PATH", "")}')


def test_run_process_captures_output(tmp_path: Path) -> None:
    """run_process возвращает код возврата и объединённый вывод."""
    bins = _make_bins(tmp_path, tmp_path / 'proj')
    ok = run_process([str(bins / 'ruff'), '--version'], tmp_path)
    assert ok.returncode == 0
    assert 'ruff fake 1.0' in ok.stdout
    _script(bins, 'failing', '#!/usr/bin/env bash\necho boom\nexit 1\n')
    bad = run_process([str(bins / 'failing')], tmp_path)
    assert bad.returncode != 0
    assert 'boom' in bad.stdout


def test_run_linter_failure_filters_conflict_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """При неудаче линтера конфликтная строка форматера не печатается."""
    bins = _make_bins(tmp_path, tmp_path / 'proj')
    _script(
        bins,
        'confruf',
        """#!/usr/bin/env bash
echo "mod.py:1:1: E501 Line too long"
echo "warning: may cause conflicts when used with the formatter: COM812"
echo "mod.py:3:5: F821 Undefined name"
exit 1
""",
    )
    bin_path = str(bins / 'confruf')
    run_linter(
        _ctx(bins, tmp_path),
        'ruff check',
        'mod.py',
        [bin_path, 'check', 'mod.py'],
        tmp_path,
    )
    out = capsys.readouterr().out
    assert 'E501 Line too long' in out
    assert 'F821 Undefined name' in out
    assert 'may cause conflicts' not in out


def test_run_linter_missing_binary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Отсутствующий исполнительный файл помечается как ошибка этапа."""
    missing = str(tmp_path / 'no_ruff')
    run_linter(
        _ctx(tmp_path / 'bin', tmp_path),
        'ruff check',
        'a.py',
        [missing, 'check', 'a.py'],
        tmp_path,
    )
    captured = capsys.readouterr()
    assert 'Инструмент не установлен' in captured.out
    assert captured.out != ''


def test_check_one_file_requires_present_tool(tmp_path: Path) -> None:
    """Без инструмента в контексте - SystemExit с ошибкой."""
    proj = tmp_path / 'proj'
    (proj / 'src').mkdir(parents=True)
    (proj / 'src' / 'mod.py').write_text('X = 1\n', encoding='utf-8')
    cfg = proj / 'pyproject.toml'
    cfg.write_text('', encoding='utf-8')
    ctx = _ctx(tmp_path / 'bin', cfg, tools={'ruff': None})
    with pytest.raises(SystemExit) as exc_info:
        check_one_file(ctx, proj, 'src/mod.py', None, annotate=False)
    assert exc_info.value.code == 1


def test_check_one_file_annotate_suffix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Подпись покрытия печатается только при annotate=True."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    ctx = _ctx(bins, proj / 'pyproject.toml')
    cov = covmod.Coverage()
    cov.metrics['src/mod.py'] = covmod.FileMetrics(50, None)
    check_one_file(ctx, proj, 'src/mod.py', cov, annotate=True)
    assert 'строки 50%' in capsys.readouterr().out
    check_one_file(ctx, proj, 'src/mod.py', cov, annotate=False)
    assert 'строки 50%' not in capsys.readouterr().out


def test_check_one_file_fix_mode_no_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """В режиме правки формат без --check, а ruff check с --fix."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    ctx = _ctx(bins, proj / 'pyproject.toml', fix=True)
    check_one_file(ctx, proj, 'src/mod.py', None, annotate=False)
    assert capsys.readouterr().out != ''
    assert ctx.reporter.failures == []


def test_check_one_file_all_fail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Маркер FAIL в файле валит все три линтера."""
    proj = _make_failing_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    ctx = _ctx(bins, proj / 'pyproject.toml')
    check_one_file(ctx, proj, 'src/mod.py', None, annotate=False)
    failures = ctx.reporter.failures
    assert len(failures) == 3
    assert 'ruff format' in failures[0]
    assert 'ruff check' in failures[1]
    assert 'mypy' in failures[2]
    out = capsys.readouterr().out
    assert 'ruf fail' in out
    assert 'mypy fail' in out


def test_run_stage_empty_bucket(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Пустой этап сообщает об отсутствии файлов и не падает."""
    ctx = _ctx(tmp_path, tmp_path)
    run_stage(ctx, 'Проверка основных файлов', {}, None, annotate=True)
    assert 'Файлы не найдены' in capsys.readouterr().out
    assert ctx.reporter.failures == []


def test_run_stage_success_and_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Успешный этап один файл, неуспешный - счётчик ошибок."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    ctx = _ctx(bins, proj / 'pyproject.toml')
    run_stage(ctx, 'Этап', {proj: ['src/mod.py']}, None, annotate=False)
    assert 'Этап завершён (1 файлов)' in capsys.readouterr().out
    failing_proj = _make_failing_proj(tmp_path)
    ctx2 = _ctx(bins, failing_proj / 'pyproject.toml')
    run_stage(
        ctx2,
        'Этап',
        {failing_proj: ['src/mod.py', 'other.py']},
        None,
        annotate=False,
    )
    assert 'с ошибками (1 из 2 файлов)' in capsys.readouterr().out


def test_run_tests_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Успешный pytest сливает метрики и удаляет временный JSON."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    targets = Targets()
    targets.add_root(proj)
    targets.add_file(proj, 'tests/test_mod.py', 'test')
    cov = run_tests(_ctx(bins, proj / 'pyproject.toml'), targets)
    out = capsys.readouterr().out
    assert 'Тесты прошли' in out
    assert 'Покрытие тестами' in out
    assert cov.totals_line() == 'строки 100%'
    assert not (proj / '.coverage_checker.json').exists()


def test_run_tests_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Падение pytest фиксируется этапом, вывод печатается, метрики слиты."""
    proj = _make_proj(tmp_path)
    (proj / 'tests' / 'test_mod.py').write_text(
        'assert FAIL\n',
        encoding='utf-8',
    )
    bins = _make_bins(tmp_path, proj)
    ctx = _ctx(bins, proj / 'pyproject.toml')
    targets = Targets()
    targets.add_root(proj)
    targets.add_file(proj, 'tests/test_mod.py', 'test')
    cov = run_tests(ctx, targets)
    assert ctx.reporter.failures == [f'pytest: {proj}']
    out = capsys.readouterr().out
    assert 'pytest-output' in out
    assert 'Покрытие тестами' not in out
    assert cov.totals_line() == 'строки 100%'


def test_run_tests_no_tests(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Без тестов pytest пропускается, покрытия нет."""
    proj = tmp_path / 'proj'
    proj.mkdir()
    bins = _make_bins(tmp_path, proj)
    targets = Targets()
    targets.add_root(proj)
    cov = run_tests(_ctx(bins, proj / 'pyproject.toml'), targets)
    assert 'Тесты не найдены' in capsys.readouterr().out
    assert cov.totals_line() is None


def test_run_tests_missing_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Отсутствие JSON-отчёта после прогона не роняет слияние."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj, with_json=False)
    targets = Targets()
    targets.add_root(proj)
    targets.add_file(proj, 'tests/test_mod.py', 'test')
    cov = run_tests(_ctx(bins, proj / 'pyproject.toml'), targets)
    assert 'Тесты прошли' in capsys.readouterr().out
    assert cov.totals_line() is None


def test_print_header_versions_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Шапка печатает версии инструментов и режимы."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    ctx = _ctx(bins, proj / 'pyproject.toml')
    print_header(ctx, LEGACY, [])
    out = capsys.readouterr().out
    assert 'Инструменты' in out
    assert 'ruff fake 1.0' in out
    assert 'Дефолтный запуск из корня dl_utils' in out


def test_print_header_targets_and_fix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """В целевом режиме и режиме правки строки шапки меняются."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    ctx = _ctx(bins, proj / 'pyproject.toml', fix=True)
    print_header(ctx, 'targets', ['a', 'b'])
    out = capsys.readouterr().out
    assert 'Целевые пути: a b' in out
    assert 'Режим правки (-f)' in out


def test_print_header_no_tools_and_default_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пропущенные инструменты отмечены, режим отчёта по умолчанию."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    monkeypatch.setattr('checker.pycheck.coverage_available', lambda: False)
    ctx = _ctx(
        bins,
        proj / 'pyproject.toml',
        tools={'ruff': None, 'mypy': None, 'nbqa': None},
    )
    print_header(ctx, 'targets', [])
    out = capsys.readouterr().out
    assert 'нет ruff' in out
    assert 'нет mypy' in out
    assert 'pytest-cov: нет (покрытие не считается)' in out
    assert 'Режим отчёта' in out


def test_main_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Полный прогон зелёного проекта успешен и показывает покрытие."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    _patch_path(monkeypatch, bins)
    assert main([str(proj)]) == 0
    out = capsys.readouterr().out
    assert 'Проверка основных файлов' in out
    assert 'строки 100%' in out


def test_main_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проваленный линтер возвращает код 1 со списком этапов."""
    proj = _make_failing_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    _patch_path(monkeypatch, bins)
    assert main([str(proj)]) == 1
    out = capsys.readouterr().out
    assert 'Проваленных этапов: 3' in out
    assert 'ruf fail' in out


def test_main_header_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Флаг -H включает печать шапки в основном потоке."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    _patch_path(monkeypatch, bins)
    assert main(['-H', str(proj)]) == 0
    assert 'Инструменты' in capsys.readouterr().out


def test_main_clean_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Флаг -c удаляет кеши инструментов в корне цели до прогона."""
    proj = _make_proj(tmp_path)
    for name in ('.mypy_cache', '.ruff_cache', '.pytest_cache'):
        cache_dir = proj / name
        cache_dir.mkdir()
        (cache_dir / 'junk').write_text('x', encoding='utf-8')
    bins = _make_bins(tmp_path, proj)
    _patch_path(monkeypatch, bins)
    assert main(['-c', str(proj)]) == 0
    for name in ('.mypy_cache', '.ruff_cache', '.pytest_cache'):
        assert not (proj / name).exists()
    assert 'Очищен кеш' in capsys.readouterr().out


def test_main_without_clean_cache_keeps_caches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без флага -c кеши инструментов не трогаются."""
    proj = _make_proj(tmp_path)
    cache_dir = proj / '.mypy_cache'
    cache_dir.mkdir()
    (cache_dir / 'junk').write_text('x', encoding='utf-8')
    bins = _make_bins(tmp_path, proj)
    _patch_path(monkeypatch, bins)
    assert main([str(proj)]) == 0
    assert (proj / '.mypy_cache').exists()
    out = capsys.readouterr().out
    assert 'Очищен кеш' not in out


def test_main_clean_cache_without_dirs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Флаг -c без кешей ничего не удаляет и не печатает сообщение."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    _patch_path(monkeypatch, bins)
    assert main(['-c', str(proj)]) == 0
    for name in ('.mypy_cache', '.ruff_cache', '.pytest_cache'):
        assert not (proj / name).exists()
    out = capsys.readouterr().out
    assert 'Очищен кеш' not in out


def test_main_legacy_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Дефолтный запуск из корня dl_utils работает по белому списку."""
    root = tmp_path / 'repo'
    for rel in ROOT_FILES:
        union = root / rel
        union.parent.mkdir(parents=True, exist_ok=True)
        union.write_text('"""Фикстура белого списка."""\n', encoding='utf-8')
    (root / 'tests').mkdir()
    (root / 'tests' / 'test_legacy.py').write_text(
        '"""Тест белого списка."""\n\n\ndef test_x() -> None:\n    assert True\n',
        encoding='utf-8',
    )
    (root / 'pyproject.toml').write_text('[project]\nname = "repo"\n', encoding='utf-8')
    _git(root)
    bins = _make_bins(tmp_path, root)
    _patch_path(monkeypatch, bins)
    monkeypatch.setenv('DLUTILS_DIR', str(root))
    monkeypatch.chdir(root)
    assert main(['-H']) == 0
    assert 'Дефолтный запуск из корня dl_utils' in capsys.readouterr().out


def test_main_fix_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Флаг -f проводит прогон в режиме автофиксов."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    _patch_path(monkeypatch, bins)
    assert main(['-f', str(proj)]) == 0


def test_main_quiet_no_files(tmp_path: Path) -> None:
    """Пустая цель в тихом режиме отдаёт код 3."""
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert main(['-q', str(empty)]) == NO_FILES


def test_main_no_files_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Пустая цель без тишины предупреждает и даёт код 0."""
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert main([str(empty)]) == 0
    assert 'Нет файлов подходящего типа' in capsys.readouterr().out


def test_main_missing_path(tmp_path: Path) -> None:
    """Отсутствующий путь завершается ошибкой с кодом 1."""
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path / 'gone')])
    assert exc_info.value.code == 1


def test_run_linter_failure_silent(tmp_path: Path) -> None:
    """Неудача линтера без вывода всё равно помечается ошибкой."""
    bins = _make_bins(tmp_path, tmp_path / 'proj')
    _script(bins, 'silent', '#!/usr/bin/env bash\nexit 1\n')
    ctx = _ctx(bins, tmp_path)
    run_linter(
        ctx,
        'ruff check',
        'mod.py',
        [str(bins / 'silent'), 'check', 'mod.py'],
        tmp_path,
    )
    assert ctx.reporter.failures == ['ruff check: mod.py']


def test_check_one_file_ipynb_uses_nbqa(tmp_path: Path) -> None:
    """Для .ipynb mypy запускается через обёртку nbqa."""
    proj = _make_proj(tmp_path)
    (proj / 'src' / 'mod.ipynb').write_text('{"cells": []}', encoding='utf-8')
    bins = _make_bins(tmp_path, proj)
    ctx = _ctx(bins, proj / 'pyproject.toml')
    check_one_file(ctx, proj, 'src/mod.ipynb', None, annotate=False)
    assert ctx.reporter.failures == []


def test_run_tests_skips_root_without_tests(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Корень цели без тестов пропускается в цикле pytest."""
    proj = _make_proj(tmp_path)
    empty = tmp_path / 'empty'
    empty.mkdir()
    bins = _make_bins(tmp_path, proj)
    targets = Targets()
    targets.add_root(proj)
    targets.add_root(empty)
    targets.add_file(proj, 'tests/test_mod.py', 'test')
    cov = run_tests(_ctx(bins, proj / 'pyproject.toml'), targets)
    assert 'Тесты прошли' in capsys.readouterr().out
    assert cov.totals_line() == 'строки 100%'


def test_run_tests_without_coverage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без pytest-cov pytest запускается без замера покрытия."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    monkeypatch.setattr('checker.pycheck.coverage_available', lambda: False)
    targets = Targets()
    targets.add_root(proj)
    targets.add_file(proj, 'tests/test_mod.py', 'test')
    cov = run_tests(_ctx(bins, proj / 'pyproject.toml'), targets)
    out = capsys.readouterr().out
    assert 'Тесты прошли' in out
    assert 'Покрытие тестами' not in out
    assert cov.totals_line() is None


def test_run_tests_failure_silent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Падение pytest без вывода помечается ошибкой и не печатает шум."""
    proj = _make_proj(tmp_path)
    (proj / 'tests' / 'test_mod.py').write_text('assert FAIL\n', encoding='utf-8')
    bins = _make_bins(tmp_path, proj, with_json=False)
    _script(bins, 'pytest', '#!/usr/bin/env bash\nexit 1\n')
    targets = Targets()
    targets.add_root(proj)
    targets.add_file(proj, 'tests/test_mod.py', 'test')
    run_ctx = _ctx(bins, proj / 'pyproject.toml')
    cov = run_tests(run_ctx, targets)
    assert run_ctx.reporter.failures == [f'pytest: {proj}']
    out = capsys.readouterr().out
    assert 'pytest-output' not in out
    assert cov.totals_line() is None


def test_run_tests_success_empty_totals(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Пустые totals после успешного pytest не дают строки покрытия."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    zero_doc = json.dumps(
        {
            'files': {},
            'totals': {
                'num_statements': 0,
                'covered_lines': 0,
                'num_branches': 0,
                'covered_branches': 0,
            },
        },
    )
    _script(
        bins,
        'pytest',
        f"""#!/usr/bin/env bash
set -u
out=""
for a in "$@"; do
  case "$a" in
    --cov-report=json:*) out="${{a#--cov-report=json:}}" ;;
  esac
done
[[ -n "$out" ]] && printf '%s' '{zero_doc}' > "$out"
exit 0
""",
    )
    targets = Targets()
    targets.add_root(proj)
    targets.add_file(proj, 'tests/test_mod.py', 'test')
    cov = run_tests(_ctx(bins, proj / 'pyproject.toml'), targets)
    out = capsys.readouterr().out
    assert 'Тесты прошли' in out
    assert 'Покрытие тестами' not in out
    assert cov.totals_line() is None


def test_main_legacy_without_git(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Дефолтный запуск без git завершается ошибкой внутри репозитория."""
    root = tmp_path / 'repo'
    root.mkdir()
    bins = tmp_path / 'bin'
    bins.mkdir()
    monkeypatch.setenv('DLUTILS_DIR', str(root))
    monkeypatch.chdir(root)
    monkeypatch.setenv('PATH', str(bins))
    with pytest.raises(SystemExit) as exc_info:
        main(['-q'])
    assert exc_info.value.code == 1
    assert 'внутри Git-репозитория' in capsys.readouterr().out


def test_main_targets_no_args_uses_cwd(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без аргументов вне корня dl_utils целью становится текущая директория."""
    proj = _make_proj(tmp_path)
    bins = _make_bins(tmp_path, proj)
    _patch_path(monkeypatch, bins)
    monkeypatch.chdir(proj)
    assert main(['-q']) == 0
    assert 'Проверка основных файлов' in capsys.readouterr().out
