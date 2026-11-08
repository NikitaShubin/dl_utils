# utils: параллельная обработка

`mpmap` — параллельный аналог `map` — и пара помощников для матричных
вычислений и перестановок.

## Обработать список в несколько процессов

```python
from pathlib import Path
from utils import mpmap

files = ['frame_01.jpg', 'frame_02.jpg', 'frame_03.jpg']


def file_size(path):
    return Path(path).stat().st_size


sizes = mpmap(
    file_size,
    files,
    num_procs=4,
    desc='Размеры файлов',
)
```

`mpmap` принимает несколько списков аргументов; они распараллеливаются
попарно, как в `Pool.starmap`:

```python
from utils import mpmap

def crop_and_save(image, box, suffix):
    ...

mpmap(crop_and_save, images, boxes, suffixes, desc='Нарезка')
```

`num_procs=0` (по умолчанию) берёт все ядра, `num_procs=1` выполняет код в
текущем процессе без `multiprocessing`. Дробное значение трактуется как доля
ядер: `num_procs=0.5`.

## Сгруппировать мелкие задачи

Для коротких функций накладные расходы на процесс больше полезной работы.
`batch_size` собирает несколько элементов в одну задачу:

```python
from utils import mpmap

areas = mpmap(compute_area, masks, batch_size=16, desc='Площади масок')
```

## Переставить аргументы для `mpmap`

`invzip` превращает список кортежей аргументов в список списков — ровно то,
что ждёт `mpmap`:

```python
from utils import invzip, mpmap

pairs = [(image_1, 'left'), (image_2, 'right')]
images, sides = invzip(pairs)

results = mpmap(process_side, images, sides)
```

## Матрица попарных значений

`apply_on_cartesian_product` применяет функцию к каждой паре элементов и
возвращает матрицу. Для симметричной функции (`symmetric=True`) считает только
верхний треугольник:

```python
import numpy as np
from utils import apply_on_cartesian_product

def iou(a, b):
    return np.sum(a & b) / np.sum(a | b)

masks = [mask_a, mask_b, mask_c]
matrix = apply_on_cartesian_product(iou, masks, symmetric=True, diag_val=1.0)
```

Если `values2` не передан, функция работает с декартовым квадратом `values1`.
Диагональ можно сразу заполнить через `diag_val`, не вызывая функцию.

## Переставить несколько списков одинаково

```python
from utils import reorder_lists

names = ['car', 'person', 'bike']
scores = [0.9, 0.6, 0.8]
order = [2, 0, 1]

names, scores = reorder_lists(order, names, scores)
# names  == ['bike', 'car', 'person']
# scores == [0.8, 0.9, 0.6]
```

## Ограничения

- Аргументы и результаты `mpmap` должны быть сериализуемы `pickle`: лямбды и
  локальные функции не подойдут, нужна функция верхнего уровня.
- Побочные эффекты в дочерних процессах не возвращаются в родительский;
  результат передавайте только через возвращаемое значение.
- `mpmap` не управляет памятью GPU: внутри процесса каждый воркер загрузит
  свою копию модели, если она создаётся в функции.
