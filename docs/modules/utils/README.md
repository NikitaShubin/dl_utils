# 🧰 utils.py — общие утилиты без DL-фреймворков

Файлы, изображения и видео, параллельная обработка, сериализация, тайминг и
диагностика. Базовый модуль, от которого зависят почти все остальные.

**Зависимости:** NumPy, OpenCV, pandas, Matplotlib, PyYAML, tqdm, IPython;
`jupyter-beeper` — опционально для `Beep`.

## 🚀 Примеры

```python
from utils import TimeIt, get_file_list

# Замерить время поиска файлов
with TimeIt('поиск файлов') as timer:
    files = get_file_list('data', '.jpg')

print(len(files), timer())
```

```python
from pathlib import Path
from utils import get_file_list, mpmap


def file_size(path):
    return Path(path).stat().st_size


# Посчитать размеры файлов параллельно
files = get_file_list('data', '.jpg')
sizes = mpmap(file_size, files, desc='Размеры')
print(sum(sizes))
```

## 🗂️ Разделы

- [Изображения и видео](media.md) — чтение кадров, паддинг, превью, плёнка,
  контрастный текст
- [Файлы и архивы](files.md) — `mkdirs`, `rmpath`, `get_file_list`, `Zipper`,
  распаковка
- [Параллельная обработка](parallel.md) — `mpmap`, матрицы, батчи, `invzip`
- [Сериализация](serialization.md) — YAML/JSON, `ManyToOne`
- [Ноутбуки и тайминг](notebook.md) — `TimeIt`, `AnnotateIt`, `Beep`,
  `DelayedInit`, `Retry`
- [Диагностика и граф](diagnostics.md) — `ImportVisitor`,
  `draw_repo_dependency_graph`, `draw_repo_dependency_graph_dot`, `obj_diff`
- [Разное](misc.md) — римские числа, индексы, случайные зёрна, `CircleInd`

## 🔧 Как устроено

`utils.py` намеренно не импортирует другие модули репозитория. Это делает его
самым нижним слоем: его можно забрать одним файлом и использовать отдельно.
Модуль большой, потому что в нём собраны независимые бытовые утилиты; при
разрезании кода на подпапки именно этот файл — первый кандидат.

Внутри есть несколько независимых групп: медиа, файлы, процессы,
сериализация, ноутбучные контексты и диагностика. Они не связаны общим
состоянием, поэтому страницы ниже можно читать отдельно.

## ⚠️ Ограничения

- Часть функций завязана на конкретные расширения OpenCV и не претендует на
  универсальный медиа-парсер.
- `mpmap` использует `multiprocessing`; объекты должны быть сериализуемы, а
  побочные эффекты в дочерних процессах не возвращаются.
- `ImReadBuffer` кеширует открытый файл; для параллельного чтения из разных
  потоков нужен отдельный экземпляр на поток.
