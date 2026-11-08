"""Тесты infra-чекера: режимы конфигов, cwd и дублирование ShellCheck."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'check-infra.sh'
CHECK = ROOT / 'check.sh'
SHELLCHECK_RC = ROOT / '.shellcheckrc'


def _make_tool(bins: Path, name: str) -> None:
    """Фейковый инструмент: пишет cwd и аргументы в журнал STUB_LOG.<имя>."""
    body = f"""#!/usr/bin/env bash
log="$STUB_LOG.{name}"
{{
    printf '%s' "$PWD"
    for a in "$@"; do printf '\\t%s' "$a"; done
    printf '\\n'
}} >> "$log"
"""
    if name == 'shellcheck':
        body += 'printf \'%s\\n\' "${SHELLCHECK_OPTS-__UNSET__}" '
        body += '>> "$STUB_LOG.shellcheck_env"\n'
    body += 'exit 0\n'
    path = bins / name
    path.write_text(body, encoding='utf-8')
    path.chmod(0o755)


def _make_bins(bins: Path, names: tuple[str, ...]) -> None:
    """Набор фейковых инструментов с общим журналом вызовов."""
    bins.mkdir()
    for name in names:
        _make_tool(bins, name)


def _make_project(tmp_path: Path) -> Path:
    """Мини-проект со всеми типами infra-файлов."""
    proj = tmp_path / 'proj'
    (proj / 'scripts').mkdir(parents=True)
    (proj / 'docs').mkdir()
    (proj / 'Dockerfile').write_text('FROM scratch\n', encoding='utf-8')
    (proj / 'docker-compose.yml').write_text(
        'services: {}\n',
        encoding='utf-8',
    )
    (proj / 'scripts' / 'run.sh').write_text(
        '#!/bin/bash\necho run\n',
        encoding='utf-8',
    )
    (proj / 'docs' / 'README.md').write_text('# Документ\n', encoding='utf-8')
    return proj


def _run(
    proj: Path,
    args: list[str],
    bins: Path,
    log: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Запуск чекера из проекта с фейковыми инструментами в PATH."""
    bash = shutil.which('bash')
    assert bash is not None
    env = os.environ.copy()
    env['PATH'] = f'{bins}{os.pathsep}{env.get("PATH", "")}'
    env['STUB_LOG'] = str(log)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(  # noqa: S603 - bash найден через which
        [bash, str(SCRIPT), *args],
        cwd=proj,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )


def _calls(path: Path) -> list[tuple[str, list[str]]]:
    """Вызовы инструмента: cwd и список аргументов."""
    calls: list[tuple[str, list[str]]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = line.split('\t')
        calls.append((fields[0], fields[1:]))
    return calls


def test_standard_mode_uses_dl_utils_configs(tmp_path: Path) -> None:
    """Стандартный режим передаёт конфиги dl_utils и работает из папки файла."""
    proj = _make_project(tmp_path)
    bins = tmp_path / 'bin'
    _make_bins(
        bins,
        ('dclint', 'hadolint', 'shellcheck', 'markdownlint'),
    )
    log = tmp_path / 'calls'
    result = _run(
        proj,
        ['-q', '.'],
        bins,
        log,
        extra_env={'SHELLCHECK_OPTS': '--exclude=SC2034'},
    )
    assert result.returncode == 0, result.stdout

    dclint_cwd, dclint_args = _calls(Path(f'{log}.dclint'))[0]
    assert dclint_cwd == str(proj)
    assert dclint_args == [
        '-c',
        str(ROOT / '.dclintrc'),
        'docker-compose.yml',
    ]

    hadolint_cwd, hadolint_args = _calls(Path(f'{log}.hadolint'))[0]
    assert hadolint_cwd == str(proj)
    assert hadolint_args == [
        '--config',
        str(ROOT / '.hadolint.yaml'),
        'Dockerfile',
    ]

    shellcheck_cwd, shellcheck_args = _calls(Path(f'{log}.shellcheck'))[0]
    assert shellcheck_cwd == str(proj / 'scripts')
    assert shellcheck_args == ['--norc', '--shell=bash', 'run.sh']
    assert Path(f'{log}.shellcheck_env').read_text(encoding='utf-8') == ('__UNSET__\n')

    markdownlint_cwd, markdownlint_args = _calls(
        Path(f'{log}.markdownlint'),
    )[0]
    assert markdownlint_cwd == str(proj / 'docs')
    assert markdownlint_args == [
        '--config',
        str(ROOT / '.markdownlint.yaml'),
        '--ignore-path',
        '/dev/null',
        'README.md',
    ]


def test_project_mode_uses_native_rules(tmp_path: Path) -> None:
    """Проектный режим не передаёт конфиги и работает из каталога вызова."""
    proj = _make_project(tmp_path)
    bins = tmp_path / 'bin'
    _make_bins(
        bins,
        ('dclint', 'hadolint', 'shellcheck', 'markdownlint'),
    )
    log = tmp_path / 'calls'
    result = _run(
        proj,
        ['-p', '-q', '.'],
        bins,
        log,
        extra_env={'SHELLCHECK_OPTS': '--exclude=SC2034'},
    )
    assert result.returncode == 0, result.stdout

    for name, expected in (
        ('dclint', ['docker-compose.yml']),
        ('hadolint', ['Dockerfile']),
        ('shellcheck', ['scripts/run.sh']),
        ('markdownlint', ['docs/README.md']),
    ):
        cwd, args = _calls(Path(f'{log}.{name}'))[0]
        assert cwd == str(proj)
        assert args == expected
    assert Path(f'{log}.shellcheck_env').read_text(encoding='utf-8') == (
        '--exclude=SC2034\n'
    )


def _parse_rc(path: Path) -> list[tuple[str, str]]:
    """Пары ключ-значение из .shellcheckrc без комментариев."""
    directives: list[tuple[str, str]] = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        key, sep, value = line.partition('=')
        assert sep, f'Нет знака = в строке .shellcheckrc: {raw}'
        directives.append((key.strip(), value.strip()))
    return directives


def _rc_to_cli(directives: list[tuple[str, str]]) -> list[str]:
    """CLI-эквивалент директив .shellcheckrc; неизвестное - ошибка теста."""
    args: list[str] = []
    for key, value in directives:
        match key:
            case 'shell':
                args.append(f'--shell={value}')
            case 'enable':
                args.append(f'--enable={value}')
            case 'disable':
                assert value != 'all', (
                    'disable=all не имеет CLI-эквивалента: измените подход '
                    'к синхронизации'
                )
                assert '-' not in value, (
                    'Диапазоны disable в .shellcheckrc нужно раскрыть '
                    'в check-infra.sh: CLI ShellCheck их не принимает'
                )
                args.append(f'--exclude={value}')
            case 'external-sources':
                assert value == 'true'
                args.append('--external-sources')
            case 'source-path':
                args.append(f'--source-path={value}')
            case 'extended-analysis':
                args.append(f'--extended-analysis={value}')
            case 'source':
                msg = 'У source= нет CLI-эквивалента: измените подход к синхронизации'
                raise AssertionError(msg)
            case _:
                msg = f'Нет CLI-эквивалента для {key}='
                raise AssertionError(msg)
    return args


def test_shellcheck_cli_matches_rc(tmp_path: Path) -> None:
    """Явные аргументы ShellCheck повторяют настройки .shellcheckrc."""
    proj = tmp_path / 'proj'
    proj.mkdir()
    (proj / 'a.sh').write_text('#!/bin/bash\necho a\n', encoding='utf-8')
    bins = tmp_path / 'bin'
    _make_bins(bins, ('shellcheck',))
    log = tmp_path / 'calls'
    result = _run(proj, ['-q', '.'], bins, log)
    assert result.returncode == 0, result.stdout
    _, args = _calls(Path(f'{log}.shellcheck'))[0]
    expected = ['--norc', *_rc_to_cli(_parse_rc(SHELLCHECK_RC)), 'a.sh']
    assert args == expected


@pytest.mark.skipif(
    shutil.which('shellcheck') is None,
    reason='shellcheck не установлен',
)
def test_real_shellcheck_rc_matches_cli(tmp_path: Path) -> None:
    """Реальный ShellCheck ведёт себя одинаково с rc-файлом и CLI-флагами.

    .shellcheckrc кладётся рядом со скриптом: --rcfile появился только в
    ShellCheck 0.10, а автообнаружение работает и в старых версиях.
    """
    shellcheck = shutil.which('shellcheck')
    assert shellcheck is not None
    script = '#!/bin/sh\n[[ -n "${x:-}" ]] && echo ok\n'

    def make(name: str, rc: str) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        (directory / 'probe.sh').write_text(script, encoding='utf-8')
        (directory / '.shellcheckrc').write_text(rc, encoding='utf-8')
        return directory

    plain = make('plain', 'shell=sh\n')
    ours = make('ours', SHELLCHECK_RC.read_text(encoding='utf-8'))
    env = os.environ.copy()
    env.pop('SHELLCHECK_OPTS', None)

    def run(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - путь проверен через which
            [shellcheck, *args, 'probe.sh'],
            cwd=directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=False,
        )

    default = run(plain)
    assert default.returncode != 0
    assert any(code in default.stdout for code in ('SC2039', 'SC3010'))
    with_rc = run(ours)
    with_cli = run(plain, '--norc', *_rc_to_cli(_parse_rc(SHELLCHECK_RC)))
    assert with_rc.returncode == with_cli.returncode == 0
    assert with_rc.stdout == with_cli.stdout


def test_check_sh_propagates_project_config(tmp_path: Path) -> None:
    """check.sh передаёт -p обоим подскриптам: оба работают из каталога вызова."""
    proj = _make_project(tmp_path)
    (proj / 'mod.py').write_text('"""Модуль."""\n', encoding='utf-8')
    (proj / 'tests').mkdir()
    (proj / 'tests' / 'test_mod.py').write_text(
        'def test_x():\n    assert True\n',
        encoding='utf-8',
    )
    bins = tmp_path / 'bin'
    _make_bins(
        bins,
        (
            'ruff',
            'mypy',
            'pytest',
            'dclint',
            'hadolint',
            'shellcheck',
            'markdownlint',
        ),
    )
    log = tmp_path / 'calls'
    bash = shutil.which('bash')
    assert bash is not None
    env = os.environ.copy()
    env['PATH'] = f'{bins}{os.pathsep}{env.get("PATH", "")}'
    env['STUB_LOG'] = str(log)
    result = subprocess.run(  # noqa: S603 - bash найден через which
        [bash, str(CHECK), '-p'],
        cwd=proj,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    assert result.returncode == 0, result.stdout

    for name in ('ruff', 'mypy'):
        for cwd, args in _calls(Path(f'{log}.{name}')):
            assert cwd == str(proj)
            assert '--config' not in args
            assert '--config-file' not in args
    for name in ('dclint', 'hadolint', 'shellcheck', 'markdownlint'):
        cwd, args = _calls(Path(f'{log}.{name}'))[0]
        assert cwd == str(proj)
        assert '-c' not in args
        assert '--config' not in args
