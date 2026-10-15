#!/usr/bin/env python3
"""Модуль для предразметки видео с использованием SAM3."""

import shutil
from pathlib import Path

import clip  # type: ignore[import-untyped]

from utils import mkdirs, rmpath

# Контрольные имена моделей SAM3 на ModelScope:
checkpoint_name2model_id = {
    'sam3.1_multiplex.pt': 'facebook/sam3.1',
}

# Имя чекпоинта по умолчанию:
default_checkpoint_name = 'sam3.1_multiplex.pt'

# Путь к модели по умолчанию:
default_model_path = Path.home() / 'models' / default_checkpoint_name

# Путь к словарю для токенизатора BPE (Byte Pair Encoding):
bpe_path = Path(clip.__file__).parent / 'bpe_simple_vocab_16e6.txt.gz'


# Ошибку отсутствия modelscope вернём только если модуль действительно понадобится:
try:
    from modelscope.hub.file_download import (  # type: ignore[import-untyped]
        model_file_download,
    )

    ModelscopeNotFoundError = None
except ModuleNotFoundError as e:
    ModelscopeNotFoundError = e


def download_sam3(path: str | Path | None = None) -> Path:
    """Использует modelscope для загрузки SAM 3.1 в заданную папку, если надо.

    Args:
        path: Путь для сохранения модели.
              Если None - используется ~/models/sam3.1_multiplex.pt.
              Если заканчивается на .pt - используется как есть.
              Иначе - добавляется /sam3.1_multiplex.pt к пути.

    Returns:
        Path: Путь к скачанному или существующему файлу модели.

    """
    # Доопределяем путь до файла:
    path = Path(path or Path.home() / 'models')
    if path.suffix != '.pt':
        path = path / default_checkpoint_name
    # Если заданный путь не имеет расширения pt, то считаем его папкой.

    # Если файл уже существует, то ничего делать не надо:
    if path.exists():
        return path

    # Если modelscope не установлен, а модели нет - возвращаем ошибку:
    if ModelscopeNotFoundError is not None:
        raise ModelscopeNotFoundError

    # Модели, которой нет в зеркале, указать нечем:
    model_id = checkpoint_name2model_id.get(path.name)
    if not model_id:
        msg = f'Неизвестная модель SAM3: "{path.name}"!'
        raise ValueError(msg)

    # Создаём временную папку внутри целевой:
    tmp_dir = path.parent / '.temp_download'
    mkdirs(tmp_dir)

    # Скачиваем файл:
    temp_file = model_file_download(
        model_id=model_id,
        file_path=path.name,
        local_dir=tmp_dir,
    )

    # Перемещаем файл в нужное место и удаляем временные файлы:
    shutil.move(temp_file, path)
    rmpath(tmp_dir)

    return path


# При автономном запуске закачиваем модель в "~/models/":
if __name__ == '__main__':
    download_sam3()
