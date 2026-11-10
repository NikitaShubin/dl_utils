# 📷 cv_utils.py — маски, рамки и метрики совпадения

Компактные объекты `Mask` и `BBox` с арифметикой, метриками IoU/Dice/Overlap,
матрицами связности и NMS. Заменяет ручные циклы `for` вокруг OpenCV и NumPy.

**Зависимости:** [`utils.py`](utils/README.md), NumPy, OpenCV, Matplotlib,
IPython, Pillow, `tqdm`. Matplotlib импортируется на уровне модуля, даже если
графики не используются.

## 🚀 Быстрый старт

Маска из NumPy-массива, её площадь и обрамляющая рамка:

```python
import numpy as np
from cv_utils import Mask

mask = Mask(np.zeros((100, 100), dtype=bool))
mask.array[20:60, 30:70] = True

print(mask.area())          # 1600
print(mask.rectangle())     # (30, 20, 70, 60) — x0, y0, x1, y1
print(mask.asbbox())        # (30, 20, 40, 40) — x, y, w, h
```

## 🎭 Маски

### Морфология и геометрия

Маски поддерживают операторы: `&`, `|`, `-`, `~`, `*` (дилатация), `/`
(эрозия), `**` (закрытие), `//` (открытие).

```python
from cv_utils import Mask

foreground = Mask(np.ones((50, 50), dtype=bool))
background = Mask(np.zeros((50, 50), dtype=bool))

combined = foreground | background
inverted = ~foreground
eroded = foreground / 3          # ядро 3×3
split = combined.split_segments()  # список связных компонент
```

### Сравнение масок

```python
mask_a = Mask(np.zeros((100, 100), dtype=bool))
mask_b = Mask(np.zeros((100, 100), dtype=bool))
mask_a.array[10:50, 10:50] = True
mask_b.array[20:60, 20:60] = True

iou, dice, overlap = mask_a.JaccardDiceOverlap_with(mask_b)
print(iou, dice, overlap)
```

### Матрица IoU для списка объектов

```python
from cv_utils import build_IoU_matrix

matrix = build_IoU_matrix([mask_a, mask_b])
print(matrix)  # 2×2, на диагонали 1.0
```

## 🔲 Рамки

```python
from cv_utils import BBox

box = BBox.from_format((30, 20, 40, 40), format='xywh')
other = BBox.from_yolo((0.5, 0.5, 0.2, 0.2), imsize=(100, 100))

print(box.area())                 # 1600
print(box & other)                # пересечение или None
print(box | other)                # объединение
print(box.Jaccard_with(other))    # IoU
```

`BBox.from_format` принимает `xywh` и `xyxy`; `BBox.from_yolo` — нормализованные
`cx, cy, w, h`.

## 🧹 NMS

Подавление дубликатов по классам и уверенности:

```python
from cv_utils import BBox, NMS

detections = [
    BBox.from_format((10, 10, 50, 50), format='xywh',
                     attribs={'label': 'car', 'confidence': 0.9}),
    BBox.from_format((12, 12, 50, 50), format='xywh',
                     attribs={'label': 'car', 'confidence': 0.7}),
    BBox.from_format((200, 200, 30, 30), format='xywh',
                     attribs={'label': 'car', 'confidence': 0.8}),
]

kept = NMS(minIoU=0.5)(detections)
print(len(kept))  # 2
```

`NMS` требует атрибуты `label` и `confidence` у каждого объекта. Объекты без
`label` попадают в отдельную группу и не подавляются между классами.

## 🎨 Визуализация

```python
import cv2
from cv_utils import Mask

img = cv2.imread('frame.jpg')
mask.draw(img, color=(0, 255, 0), alpha=0.4)
cv2.imshow('mask', img)
```

`Mask.show()` и `BBox.show()` используют Matplotlib и удобны в ноутбуке;
`draw()` возвращает изображение и не показывает его.

## 📦 Что внутри

- `Mask` хранит двумерный массив, лениво считает площадь и рамку, умеет
  разбивать маску на связные сегменты и аппроксимировать эллипсами.
- `BBox` хранит `xyxy` и умеет пересекаться, объединяться и считать метрики.
- `build_*_matrix*` строят матрицы попарных метрик, в том числе параллельно
  через `mpmap`.
- `split_by_attrib` и `sort_by_attrib` — вспомогательные группировка и
  сортировка объектов по атрибутам; на них же опирается `NMS`.

## ⚠️ Ограничения

- `Mask` работает с двумерными массивами; цветные изображения — через `BBox`
  или отдельные каналы.
- `NMS` не реализует soft-NMS и не учитывает вложенные объекты разных классов.
- Метрики считаются по площади; для разреженных масок это честно, но дороже,
  чем сравнение рамок.
