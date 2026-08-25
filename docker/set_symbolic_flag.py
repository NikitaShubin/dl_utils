#!/usr/bin/env python3
"""set_symbolic_flag.py - нейтрализация конфликта двойных LLVM/MLIR.

Колёса triton и tensorflow статически линкуют собственные копии
LLVM/MLIR и экспортируют её символы наружу. Кто из них загрузился
в процесс первым - занимает глобальное пространство имён, после чего
статические инициализаторы второго исполняют чужой код со своими
структурами данных: сегфолт или дубли реестров опций CommandLine.

Лечение - штатный ELF-флаг DF_SYMBOLIC в секции .dynamic: «собственные
ссылки разрешать в себя самого прежде глобального пространства имён».
Одна запись на библиотеку вместо правки десятков тысяч символов;
порядок загрузки перестаёт иметь значение в обе стороны.

Место под запись берётся у хинта DT_RELACOUNT - он используется только
утилитой prelink, отсутствующей в образе. Если DT_FLAGS уже существует,
бит просто OR-ится. Идемпотентно: установленный бит повторным прогоном
не меняется, а починенные апстримом колёса флаг делают ноу-опом.

Запуск: set_symbolic_flag.py [--dry-run]
Цели ищутся в site-packages по маскам TARGETS.
"""

# Диагностический вывод - контракт скрипта, а не отладочный мусор:
# ruff: noqa: T201

import site
import struct
import sys
from pathlib import Path
from typing import BinaryIO

PT_DYNAMIC = 2
DT_FLAGS = 30
DT_RELACOUNT = 0x6FFFFFF9
DF_SYMBOLIC = 0x2

# Меньше двух целей - конфликт не имеет смысла:
MIN_TARGETS = 2

# Тяжёлые носители встроенного LLVM внутри site-packages:
TARGETS = (
    'triton/_C/libtriton*.so*',
    'tensorflow/libtensorflow_framework*.so*',
)

PHDR_FMT = '<IIQQQQ'  # p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz
DYN_FMT = '<qQ'  # d_tag, d_un
DYN_SIZE = struct.calcsize(DYN_FMT)


def discover() -> list[Path]:
    """Пути целей по маскам TARGETS во всех site-packages окружения."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in sorted({Path(p) for p in site.getsitepackages()}):
        for pattern in TARGETS:
            for path in root.glob(pattern):
                real = path.resolve()
                if real not in seen:
                    seen.add(real)
                    found.append(path)
    return found


def dynamic_span(f: BinaryIO) -> tuple[int, int]:
    """(смещение, размер) сегмента PT_DYNAMIC."""
    f.seek(0x20)
    (e_phoff,) = struct.unpack('<Q', f.read(8))
    f.seek(0x36)
    e_phentsize, e_phnum = struct.unpack('<HH', f.read(4))
    phdr_size = struct.calcsize(PHDR_FMT)
    for i in range(e_phnum):
        f.seek(e_phoff + i * e_phentsize)
        fields = struct.unpack(PHDR_FMT, f.read(phdr_size))
        p_type, _, p_offset, _, _, p_filesz = fields
        if p_type == PT_DYNAMIC:
            return p_offset, p_filesz
    msg = 'нет сегмента PT_DYNAMIC'
    raise ValueError(msg)


def scan_slots(
    f: BinaryIO,
    start: int,
    size: int,
) -> tuple[tuple[int, int] | None, int | None]:
    """Смещения записей DT_FLAGS и донорского DT_RELACOUNT."""
    flags_pos: tuple[int, int] | None = None
    donor_pos: int | None = None
    f.seek(start)
    for pos in range(start, start + size, DYN_SIZE):
        tag, val = struct.unpack(DYN_FMT, f.read(DYN_SIZE))
        if tag == DT_FLAGS:
            flags_pos = pos, val
        elif tag == DT_RELACOUNT and val:
            donor_pos = pos
    return flags_pos, donor_pos


def patch(path: Path, *, dry: bool = False) -> bool:
    """Ставит DF_SYMBOLIC в .dynamic файла; False, если бит уже стоял."""
    with path.open('rb' if dry else 'r+b') as f:
        start, size = dynamic_span(f)
        flags, donor = scan_slots(f, start, size)

        if flags:
            pos, val = flags
            if val & DF_SYMBOLIC:
                print(f'[skip] {path.name}: флаг уже стоит')
                return False
            new_val = val | DF_SYMBOLIC
        elif donor:
            pos, new_val = donor, DF_SYMBOLIC
        else:
            msg = f'{path.name}: нет ни DT_FLAGS, ни донорского RELACOUNT'
            raise RuntimeError(msg)

        if dry:
            print(f'[dry] {path.name}: будет записан DF_SYMBOLIC')
            return True
        f.seek(pos)
        f.write(struct.pack(DYN_FMT, DT_FLAGS, new_val))
        print(f'[ok] {path.name}: DF_SYMBOLIC установлен')
        return True


def main(argv: list[str]) -> int:
    """Точка входа: патчит все найденные цели; 1 при любом сбое."""
    dry = '--dry-run' in argv
    paths = discover()
    if len(paths) < MIN_TARGETS:
        sys.exit('целей меньше двух, конфликт нечего решать')
    failed = False
    for path in paths:
        try:
            patch(path, dry=dry)
        except (OSError, ValueError, RuntimeError, struct.error) as exc:
            print(f'[err] {path.name}: {exc}', file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
