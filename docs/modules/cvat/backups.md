# cvat: бекапы, XML и неразмеченные данные

Парсинг распакованных CVAT-бекапов, сохранение разметки в XML и подготовка
задач без разметки.

## Прочитать распакованный бекап

```text
backups/
    Project_1/
        Task_1/
            task.json
            annotations.json
            data/
                video.mp4
```

```python
from cvat import cvat_backups2tasks

tasks = cvat_backups2tasks('backups')
for task in tasks:
    for df, file, true_frames in task:
        print(len(df), file, len(true_frames))
```

Функция читает все задачи, удаляет дубли, интерполирует контуры и делит
последовательности по меткам `scene-start` / `scene-finish`. Результат можно
передать в [`seg.save_tasks2segmentation_dataset`](../seg.md).

Только парсинг, без постобработки:

```python
from cvat import cvat_backups2raw_tasks

raw_tasks = cvat_backups2raw_tasks('backups', desc='Парсинг')
```

## Прочитать одну задачу

```python
from cvat import cvat_backup_task_dir2task

task = cvat_backup_task_dir2task('backups/Project_1/Task_1')
print(len(task))  # число подзадач
```

Вернётся список кортежей `(df, file, true_frames)`.

## Сохранить подзадачу в XML

```python
from cvat import subtask2xml

subtask = (df, 'data/video.mp4', {0: 0, 1: 1})
subtask2xml(subtask, 'exports/annotation.xml')
```

XML можно импортировать в CVAT как разметку задачи. Если `xml_file` не
указан, функция возвращает XML-элемент и ничего не пишет на диск.

## Создать задачу без разметки

```python
from cvat import file2unlabeled_subtask

subtask = file2unlabeled_subtask('data/images')
df, file, true_frames = subtask
print(len(true_frames))
```

Если передан видеофайл, будет один кадр на каждое декодируемое изображение;
если папка — все изображения по порядку. По умолчанию число кадров
оценивается по метаданным; `check_frames=True` декодирует их, чтобы проверить
читаемость. В возвращаемой подзадаче датафрейм равен `None`.

## Собрать задачи из папки

```python
from cvat import dir2unlabeled_tasks

tasks = dir2unlabeled_tasks('data/unlabeled', check_frames=True)
```

Каждый видеофайл становится отдельной задачей, изображения группируются по
папкам. `check_frames=True` проверяет, что кадры действительно читаются.

## Разметка Яндекс Крауда

```python
from cvat import yandex2subtask

subtask = yandex2subtask('crowd/annotation.json', 'data/video.mp4')
```

Функция переводит JSON Яндекс Крауда в формат CVAT-подзадачи.

## Имя проекта и задачи

```python
from cvat import cvat_backup_task_dir2info

info = cvat_backup_task_dir2info('backups/Project_1/Task_1')
print(info['proj_name'], info['task_name'])
```

Функция понимает пути к папке задачи, папке `data` и к файлу внутри неё.

## Связанные файлы

```python
from cvat import get_related_files

related = get_related_files('data/task/images', as_manifest=True)
```

CVAT позволяет прикладывать к изображению дополнительные файлы (карты
глубины, маски). Функция собирает словарь или manifest-список таких связей.

## Ограничения

- `cvat_backups2tasks` ожидает уже распакованные бекапы; распаковку выполняет
  `cvat_srv`.
- `cvat_backups2tasks` включает интерполяцию и разбиение по сценам. Для
  чистых данных используйте `cvat_backups2raw_tasks`.
- `subtask2xml` не копирует изображения и видео: XML содержит только разметку.
- `dir2unlabeled_tasks` сортирует изображения по именам; числовой порядок
  (`1, 2, 10`) не гарантируется.
- `yandex2subtask` поддерживает только формат Crowd — другие форматы
  импортируйте через CVAT.
