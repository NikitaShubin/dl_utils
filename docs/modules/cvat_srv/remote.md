# cvat_srv: серверные объекты

Навигация по CVAT, создание проектов и задач, бекапы, экспорт и импорт
разметки. Все операции идут через сервер, локальных копий здесь нет.

## Подключиться и найти объект

```python
from cvat_srv import CVATSRV

srv = CVATSRV(
    host='http://localhost:8080',
    username='admin',
    password='admin',
)

project = srv['Project 1']
task = project['Task 1']
job = task[0]  # первая подзадача; подзадачи адресуются по номеру
```

`CVATSRV` можно создать без аргументов — тогда адрес, логин и пароль
запрашиваются в консоли. По URL объекта работает отдельный вход:

```python
task = srv.from_url('http://localhost:8080/tasks/42')
```

## Пройти по иерархии

```python
print(srv.keys())          # имена проектов
print(project.keys())      # имена задач проекта
print(task.keys())         # номера подзадач задачи

print(task.parent().name)  # Project 1
print(len(project))        # число задач в проекте
```

Имена — это реальные имена объектов в CVAT. Если в проекте две задачи с
одинаковым именем, `project['Task 1']` вызовет `IndexError`.

## Создать проект и задачу

```python
project = srv.new('Project 2', ['car', 'person'])

task = project.new(
    'Task 2',
    'data/video.mp4',
    annotation_file='annotations.xml',
)
```

Если разметки нет, `annotation_file` можно не передавать. Задача создаётся с
метками проекта; видео должно быть единственным файлом, изображения можно
передавать списком.

## Скачать бекап

```python
task_path = task.backup('./backups', desc='Бекап задачи')
project_path = project.backup('./backups', desc='Бекап проекта')
```

Функция возвращает путь до архива. Если `path` оканчивается на `.zip`, это
считается именем файла; иначе имя собирается как `<path>/<имя объекта>.zip`.
Для набора объектов используйте список имён:

```python
paths = project.backup('./backups', name=['Task 1', 'Task 2'])
```

## Восстановить из бекапа

```python
new_project = srv.restore('./backups/Project 1.zip')
new_task = srv['Project 1'].restore('./backups/Task 1.zip')
```

`srv.restore` создаёт проект, `project.restore` — задачу внутри проекта.
Массовые операции:

```python
project.backup_all('./backups')
restored = srv.restore_all('./backups', return_id_only=False)
```

`restore_all` принимает папку с архивами или список путей и возвращает
объекты, если `return_id_only=False`.

## Экспорт и импорт разметки

```python
task.get_annotations('exports/Task 1.xml', format_name='CVAT for video 1.1')
task.set_annotations('exports/Task 1.xml', format_name='CVAT 1.1')
```

`get_annotations` не перезаписывает существующий файл и сам создаёт папку.
Если указать `.zip`, архив сохранится целиком; иначе из него извлекается
`annotations.*`.

Полный экспорт датасета вместе с изображениями:

```python
task.export('exports/Task 1.zip', format='YOLO 1.1', include_images=True)
```

## Описание задачи

```python
task.guide = 'Размечайте только автомобили и пешеходов.'
print(task.guide)
```

Гайд хранится в Markdown. Вместо строки можно передать путь к файлу — он
прочитается при установке.

## Прочитать задачи без локального редактора

`ReadTasks` скачивает бекапы, распаковывает их и возвращает сырые задачи. По
завершении контекста временные папки удаляются:

```python
from cvat_srv import ReadTasks

with ReadTasks('./tmp/cvat_backup', './tmp/cvat_extract', project) as raw_tasks:
    for df, file, true_frames in raw_tasks:
        print(len(df), file)
```

Если вызвать объект без `with`, удаления не будет — это удобно, когда
результат нужен после завершения чтения.

## Удалить объект

```python
task.delete()
```

Удаление необратимо и выполняется сразу на сервере. Перед ним стоит сделать
бекап.

## Ограничения

- `CVATSRV` делает сетевые запросы уже в конструкторе; проверьте адрес и
  учётные данные до создания объекта.
- `backup` и `restore` работают с архивами CVAT; формат экспорта разметки
  задаётся отдельно через `format_name`.
- `ReadTasks` удаляет `backup_path` и `extract_path` целиком, не только свои
  файлы. Не указывайте там ценные папки.
- Восстановление задачи возможно только на уровне сервера или проекта;
  `task.restore(...)` не поддерживается.
