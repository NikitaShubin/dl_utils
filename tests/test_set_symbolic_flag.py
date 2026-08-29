"""Тесты docker/set_symbolic_flag.py: поиск целей и патч ELF-флагов."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import pytest

from docker.set_symbolic_flag import (
    DF_SYMBOLIC,
    DT_FLAGS,
    DT_RELACOUNT,
    DYN_FMT,
    DYN_SIZE,
    PHDR_FMT,
    PT_DYNAMIC,
    discover,
    dynamic_span,
    main,
    patch,
    scan_slots,
)

if TYPE_CHECKING:
    from pathlib import Path


def _elf(
    path: Path,
    entries: list[tuple[int, int]],
    *,
    has_dynamic: bool = True,
) -> Path:
    """Файл-заглушка ELF64 с одним сегментом PT_DYNAMIC и записями .dynamic."""
    phoff = 0x40
    p_offset = 0x80
    header = bytearray(0x80)
    struct.pack_into('<Q', header, 0x20, phoff)
    struct.pack_into('<HH', header, 0x36, struct.calcsize(PHDR_FMT), 1)
    dynamic = b''.join(struct.pack(DYN_FMT, tag, val) for tag, val in entries)
    phdr_type = PT_DYNAMIC if has_dynamic else 0
    phdr = struct.pack(PHDR_FMT, phdr_type, 0, p_offset, 0, 0, len(dynamic))
    pad = p_offset - 0x40 - len(phdr)
    path.write_bytes(bytes(header[:0x40]) + phdr + b'\x00' * pad + dynamic)
    return path


def _tool(path: Path, name: str) -> Path:
    """Цель в layout site-packages: путь к .so-файлу с пустым телом."""
    sub = path / name
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_bytes(b'x')
    return sub


def test_dynamic_span_and_missing(tmp_path: Path) -> None:
    """Смещение PT_DYNAMIC находится, его отсутствие - ошибка."""
    target = _elf(tmp_path / 'a.so', [(DT_FLAGS, 0)])
    with target.open('rb') as f:
        assert dynamic_span(f) == (0x80, DYN_SIZE)
    no_dynamic = _elf(tmp_path / 'no.so', [], has_dynamic=False)
    with no_dynamic.open('rb') as f, pytest.raises(ValueError, match='PT_DYNAMIC'):
        dynamic_span(f)


def test_scan_slots_flags_and_donor(tmp_path: Path) -> None:
    """Находятся и DT_FLAGS, и донорский DT_RELACOUNT."""
    target = _elf(tmp_path / 'a.so', [(DT_FLAGS, 0x0), (DT_RELACOUNT, 0x1)])
    with target.open('rb') as f:
        flags, donor = scan_slots(f, 0x80, 2 * DYN_SIZE)
    assert flags == (0x80, 0x0)
    assert donor == 0x80 + DYN_SIZE


def test_patch_skip_when_flag_already_set(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Установленный бит - ноу-оп: skip, False, файл не меняется."""
    target = _elf(tmp_path / 'a.so', [(DT_FLAGS, DF_SYMBOLIC)])
    before = target.read_bytes()
    assert not patch(target)
    assert target.read_bytes() == before
    assert '[skip]' in capsys.readouterr().out


def test_patch_dry_run_does_not_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """В сухом режиме запись не выполняется."""
    target = _elf(tmp_path / 'a.so', [(DT_RELACOUNT, 0x1)])
    before = target.read_bytes()
    assert patch(target, dry=True)
    assert target.read_bytes() == before
    assert '[dry]' in capsys.readouterr().out


def test_patch_donor_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Без DT_FLAGS бит пишется в ячейку донорского RELACOUNT."""
    target = _elf(tmp_path / 'a.so', [(DT_RELACOUNT, 0x1)])
    assert patch(target)
    assert '[ok]' in capsys.readouterr().out
    with target.open('rb') as f:
        f.seek(0x80)
        assert struct.unpack(DYN_FMT, f.read(DYN_SIZE)) == (DT_FLAGS, DF_SYMBOLIC)


def test_patch_flags_or_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Существующие флаги OR-ятся с DF_SYMBOLIC без замены тега."""
    target = _elf(
        tmp_path / 'a.so',
        [(DT_FLAGS, 0x0), (DT_RELACOUNT, 0x1)],
    )
    assert patch(target)
    capsys.readouterr()
    with target.open('rb') as f:
        f.seek(0x80)
        assert struct.unpack(DYN_FMT, f.read(DYN_SIZE)) == (DT_FLAGS, DF_SYMBOLIC)


def test_patch_no_slot_raises(tmp_path: Path) -> None:
    """Нет ни флагов, ни донора - ошибка RuntimeError."""
    target = _elf(tmp_path / 'a.so', [])
    with pytest.raises(RuntimeError, match='RELACOUNT'):
        patch(target)


def test_discover_deduplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dupe-по symlink схлопывается, маски находят оба носителя."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    real = _tool(first, 'triton/_C/libtriton.so')
    _tool(second, 'tensorflow/libtensorflow_framework.so.2')
    linked = second / 'triton/_C/libtriton.so'
    linked.parent.mkdir(parents=True, exist_ok=True)
    linked.symlink_to(real)
    monkeypatch.setattr(
        'docker.set_symbolic_flag.site.getsitepackages',
        lambda: [str(first), str(second)],
    )
    found = discover()
    assert len(found) == 2
    names = {p.name for p in found}
    assert names == {'libtriton.so', 'libtensorflow_framework.so.2'}


def test_main_less_than_two_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Меньше двух целей - конфликт нечего решать, выход с сообщением."""
    monkeypatch.setattr('docker.set_symbolic_flag.discover', list)
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert 'меньше двух' in str(exc_info.value.code)


def test_main_dry_run_all_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Сухой прогон по всем целям завершается нулём без записи."""
    targets = [
        _elf(tmp_path / 'a.so', [(DT_RELACOUNT, 0x1)]),
        _elf(tmp_path / 'b.so', [(DT_FLAGS, 0x0)]),
    ]
    monkeypatch.setattr('docker.set_symbolic_flag.discover', lambda: targets)
    assert main(['--dry-run']) == 0
    assert '[dry]' in capsys.readouterr().out


def test_main_has_patched_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Боевой прогон патчит все цели и возвращает 0."""
    targets = [
        _elf(tmp_path / 'a.so', [(DT_RELACOUNT, 0x1)]),
        _elf(tmp_path / 'b.so', [(DT_FLAGS, 0x0)]),
    ]
    monkeypatch.setattr('docker.set_symbolic_flag.discover', lambda: targets)
    assert main([]) == 0
    out = capsys.readouterr()
    assert '[ok]' in out.out
    assert '[err]' not in out.err


def test_main_reports_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Сбой одной цели отмечается в stderr, код возврата 1."""
    good = _elf(tmp_path / 'good.so', [(DT_RELACOUNT, 0x1)])
    bad = _elf(tmp_path / 'bad.so', [], has_dynamic=False)
    monkeypatch.setattr('docker.set_symbolic_flag.discover', lambda: [good, bad])
    assert main([]) == 1
    captured = capsys.readouterr()
    assert '[err]' in captured.err
    assert '[ok]' in captured.out
