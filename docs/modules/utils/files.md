# utils: файлы и архивы

Создание и удаление путей, поиск файлов, упаковка и распаковка архивов.

## Создать и удалить путь

```python
from utils import mkdirs, rmpath

mkdirs('results/run_01')   # создаст все недостающие папки
rmpath('results/run_01')   # удалит папку со всем содержимым
rmpath(['a.tmp', 'b.tmp']) # список путей удаляется последовательно
```

Обе функции не падают, если путь уже существует или уже удалён.

## Найти файлы

```python
from utils import get_file_list

images = get_file_list('data', ['.jpg', '.png'])
videos = get_file_list('data', ['.mp4', '.avi'], recurcive=False)
dirs = get_file_list('data', 'dir')
```

По умолчанию поиск рекурсивный. Расширения можно передать строкой, списком
или множеством; регистр не важен. `'.*'` или пустой список означают «все
файлы».

## Архивировать и распаковать

```python
from utils import Zipper

archive = Zipper.compress('dataset', 'dataset.zip')
restored = Zipper.extract('dataset.zip', 'dataset_copy')
```

`Zipper` также работает как объект с заранее заданными путями:

```python
from utils import Zipper

zipper = Zipper(unzipped='dataset', zipped='dataset.zip')
zipper.compress()
zipper.extract()
```

`remove_source=True` удаляет исходник после успешной операции,
`rewrite_target=True` разрешает перезаписать цель.

## Распаковать папку с архивами

```python
from utils import unzip_dir

unzipped_dirs = unzip_dir('backups', 'unzipped', desc='Распаковка бекапов')
```

Каждый `*.zip` распаковывается в подпапку с именем архива. Функция не
рекурсивна: смотрит только файлы верхнего уровня.

## Имена файлов

```python
from utils import str2filename, filename2str, split_dir_name_ext

safe = str2filename('Проект: задача №1')  # 'Proekt_zadacha_1' (пример)
original = filename2str(safe)
path, name, ext = split_dir_name_ext('data/video.mp4')
```

`str2filename` заменяет недопустимые символы и сохраняет возможность обратного
преобразования через `filename2str`.

## Временные и пустые папки

```python
from utils import get_empty_dir, first_existed_path

tmp = get_empty_dir()                    # новая временная папка
cache = get_empty_dir('cache', clear=True)  # очистить, если не пуста
model = first_existed_path(['/mnt/models', './models', 'models'])
```

`get_empty_dir` по умолчанию падает, если папка не пуста; `clear=True`
разрешает очистку.

## Ограничения

- `rmpath` удаляет без подтверждения; для ценных путей проверяйте путь до
  вызова.
- `Zipper` не сохраняет пустые папки и символические ссылки так, как это
  делает `tar`.
- `get_file_list` возвращает пути в порядке `os.listdir`; сортируйте результат,
  если порядок важен.
