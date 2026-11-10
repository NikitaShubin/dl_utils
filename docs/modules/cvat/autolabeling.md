# 🤖 cvat: авторазметка

Применение модели к готовой CVAT-задаче или к распакованному бекапу. Модель —
любой вызываемый объект, который принимает RGB-кадр и возвращает датафрейм в
формате CVAT.

## 🚀 Примеры

```python
# Применить модель к готовой задаче
from cvat import tasks_auto_annottation
from gdinoal import GDino

gdino = GDino(prompt2label={'Person': 'person'})
tasks = tasks_auto_annottation(tasks, gdino.img2df, label='person')
```

```python
# Разметить весь бекап: по XML на каждую задачу
from cvat import cvat_backup_dir2auto_annotation_xmls
from gdinoal import GDino

gdino = GDino(prompt2label={'Person': 'person'})
cvat_backup_dir2auto_annotation_xmls(
    'backups/Project_1',
    gdino.img2df,
    label='person',
    xml_dir='exports/auto',
)
```

## 🎯 Разметка задач

### Разметить задачу

```python
from cvat import tasks_auto_annottation
from gdinoal import GDino

gdino = GDino(prompt2label={'Person': 'person'})

tasks = tasks_auto_annottation(
    tasks,
    gdino.img2df,
    label='person',
    store_prev_annotation=True,
    desc='Авторазметка',
)
```

`store_prev_annotation=True` сохраняет исходную разметку и добавляет новую;
`False` заменяет её результатом модели. Параметр `label` заполняет пропущенные
метки в ответе модели.

### Постобработка подзадач

```python
from cvat import hide_skipped_objects_in_subtask

tasks = tasks_auto_annottation(
    tasks,
    gdino.img2df,
    label='person',
    postprocess_filters=[hide_skipped_objects_in_subtask],
)
```

Фильтр получает и возвращает подзадачу `(df, file, true_frames)`. Так можно
скрыть объекты в кадрах, где модель их не нашла.

### Сброс модели

Если у `img2df` есть метод `reset`, он вызывается до и после обработки. Это
важно для трекеров и моделей с состоянием.

## 💾 Разметка бекапов

### Разметить одну задачу из бекапа

```python
from cvat import cvat_backup_task_dir2auto_annotation_xml

cvat_backup_task_dir2auto_annotation_xml(
    'backups/Project_1/Task_1',
    gdino.img2df,
    label='person',
    xml_file='exports/Task_1_auto.xml',
    desc='Авторазметка задачи',
)
```

Функция читает распакованную задачу, применяет модель и сохраняет XML,
совместимый с CVAT.

### Разметить весь бекап

```python
from cvat import cvat_backup_dir2auto_annotation_xmls

cvat_backup_dir2auto_annotation_xmls(
    'backups/Project_1',
    gdino.img2df,
    label='person',
    xml_dir='exports/auto',
    num_procs=2,
)
```

Для каждой задачи создаётся отдельный XML в `xml_dir`.

## ⚠️ Ограничения

- Модель должна возвращать датафрейм с колонками CVAT или `None`; строки без
  метки получат `label` из аргумента.
- `cvat_backup_task_dir2auto_annotation_xml` работает только с задачами из
  одной подзадачи.
- При `num_procs > 1` модель копируется в каждый процесс; GPU-память может
  закончиться.
- Авторазметка не проверяет геометрию: контуры могут выходить за границы
  кадра. `autofix_df` решает другую задачу — убирает смену класса внутри
  трека.
- XML не содержит исходных файлов; для импорта в CVAT задача должна уже
  существовать.
