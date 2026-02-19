#!/usr/bin/env python3
"""Модуль для предразметки видео с использованием SAM3."""

import shutil
from pathlib import Path

# Ошибку отсутствия modelscope вернём только если он действительно понадобится:
try:
    from modelscope.hub.file_download import (  # type: ignore[import-untyped]
        model_file_download,
    )
    ModelscopeNotFoundError = None
except ModuleNotFoundError as e:
    ModelscopeNotFoundError = e


def download_sam3(path: str | Path | None = None) -> Path:
    """Использует modelscope для загрузки SAM3 в заданную папку, если надо.

    Args:
        path: Путь для сохранения модели.
              Если None - используется ~/models/sam3.pt.
              Если заканчивается на .pt - используется как есть.
              Иначе - добавляется /sam3.pt к пути.

    Returns:
        Path: Путь к скачанному или существующему файлу модели.

    """
    # Доопределяем путь до файла:
    path = Path(path or Path.home() / 'models')
    if path.suffix != '.pt':
        path = path / 'sam3.pt'
    # Если заданный путь не имеет расширения pt, то считаем его папкой.

    # Если файл уже существует, то ничего делать не надо:
    if path.exists():
        return path

    # Если modelscope не установлен, а модели нет - возвращаем ошибку:
    if ModelscopeNotFoundError is not None:
        raise ModelscopeNotFoundError

    # Создаём временную папку внутри целевой:
    tmp_dir = path.parent / '.temp_download'
    tmp_dir.mkdir(parents=True, exist_ok=False)
    tmp_name = 'sam3.pt'

    # Скачиваем файл:
    temp_file = model_file_download(
        model_id='facebook/sam3',
        file_path=tmp_name,
        local_dir=tmp_dir,
    )

    # Перемещаем файл в нужное место и удаляем временные файлы:
    shutil.move(temp_file, path)
    shutil.rmtree(tmp_dir)

    return path


# При автономном запуске закачиваем модель в "~/models/":
if __name__ == '__main__':
    download_sam3()
