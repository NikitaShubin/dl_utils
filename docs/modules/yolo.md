# 📦 yolo.py — экспорт задач в YOLO-датасет

Разбиение задач на train/val/test, статистика классов и сохранение выборок в
формате YOLO для детекции и сегментации.

**Зависимости:** `pandas`, `numpy`, `opencv-python`, `tqdm`, `labels.py`,
`cvat.py`, `cv_utils.py`, `utils.py`.

## 🚀 Примеры

```python
# Собрать статистику и разбить задачи
from labels import LabelsConvertor
from yolo import sources2statistic_and_train_val_test_tasks

converter = LabelsConvertor('labels.xlsx', 'superlabels.xlsx')
train_tasks, val_tasks, test_tasks = (
    sources2statistic_and_train_val_test_tasks(
        {'cvat': [project_1_tasks]}, 'dataset/yolo', converter,
        val_size=0.2, test_size=0.1,
    )
)
```

```python
# Сохранить датасет в формате YOLO
from yolo import tasks2yolo

tasks2yolo(
    mode='box',
    tasks=train_tasks,
    labels_convertor=converter,
    images_dir='dataset/yolo/images/train',
    lablels_dir='dataset/yolo/labels/train',
)
```

```python
# Нарисовать метки на кадре
from yolo import YOLOLabels

labels = YOLOLabels('labels/train/00000001.txt', imsize=(1080, 1920))
preview = labels.draw_labels(image)
```

## 📊 Статистика

### Собрать статистику и разбить задачи

```python
from labels import LabelsConvertor
from yolo import sources2statistic_and_train_val_test_tasks

converter = LabelsConvertor('labels.xlsx', 'superlabels.xlsx')

train_tasks, val_tasks, test_tasks = (
    sources2statistic_and_train_val_test_tasks(
        {'cvat': [project_1_tasks, project_2_tasks]},
        'dataset/yolo',
        converter,
        val_size=0.2,
        test_size=0.1,
    )
)
```

`source_name2tasks` — словарь «имя источника → список задач». Для источника
`cvat` деление идёт по именам бекапов: задачи с `test` в имени попадают в
проверочную выборку. Статистика сохраняется в `dataset/yolo/statistics`
(CSV, Excel и PNG).

### Посчитать статистику отдельно

```python
from yolo import tasks2statistic

stat = tasks2statistic(tasks, 'cvat', converter)
print(stat.head())
```

`df2statistic` делает то же для одного датафрейма. `superclass_statistic`
схлопывает статистику классов в суперклассы.

## 💾 Датасет

### Сохранить YOLO-датасет

```python
from yolo import tasks2yolo

tasks2yolo(
    mode='box',
    tasks=train_tasks,
    labels_convertor=converter,
    images_dir='dataset/yolo/images/train',
    lablels_dir='dataset/yolo/labels/train',
    preview_dir='dataset/yolo/preview/train',
    scale=1,
    max_imsize=(1080, 1920),
    desc='Сохранение train',
)
```

`mode='box'` сохраняет прямоугольники, `mode='seg'` — полигоны. Для больших
кадров изображение разрезается на части по `max_imsize`. Параметр `scale`
может быть числом, списком масштабов или множеством, из которого масштаб
выбирается случайно для каждого кадра.

### Сгенерировать `data.yaml`

```python
from yolo import gen_yaml

superclasses = {
    index: name for name, index in converter.superlabels2superinds.items()
}

gen_yaml(
    'dataset/yolo/data.yaml',
    'dataset/yolo',
    'dataset/yolo/images/train',
    'dataset/yolo/images/val',
    'dataset/yolo/images/test',
    superclasses,
)
```

Файл содержит абсолютный путь к датасету, относительные пути к выборкам и
словарь `index → имя класса`.

## 🎨 Визуализация

### Нарисовать метки

```python
from yolo import YOLOLabels

labels = YOLOLabels('dataset/yolo/labels/train/00000001.txt', imsize=(1080, 1920))
preview = labels.draw_labels(image)
```

`YOLOLabels` читает один `.txt`-файл разметки и умеет наносить рамки или
полигоны на изображение.

## ⚠️ Ограничения

- Датасет YOLO не содержит разбиения на сцены: каждый кадр — отдельный
  семпл, поэтому соседние кадры одного видео могут попасть в train и val.
- `tasks2yolo` использует `mpmap` и параллелит задачи; при большом числе
  процессов память быстро заканчивается.
- Имена классов в `data.yaml` берутся из конвертора; метки, которых нет в
  таблице суперклассов, могут получить индекс `-1` и не попасть в файл.
- Для `mode='seg'` контуры должны быть полигонами; прямоугольники
  преобразуются, но теряют форму объекта.
- `gen_yaml` не создаёт папки: все пути должны существовать заранее.
