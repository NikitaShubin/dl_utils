# 🎯 ul_utils.py — Ultralytics YOLO

Обёртка над моделями Ultralytics: инференс, трекинг, постобработка и
конвертация результатов в CVAT-датафрейм.

**Зависимости:** `ultralytics`, `cv_utils.py`, `cvat.py`, `video_utils.py`,
`utils.py`, `numpy`, `pandas`, `tqdm`.

## 🚀 Примеры

```python
# Разметить изображение
from ul_utils import UltralyticsModel

model = UltralyticsModel('yolo11n.pt', mode='preannotation')
df = model.img2df(image)
```

```python
# Показать превью
model = UltralyticsModel('yolo11n-seg.pt', mode='preview')
preview = model.draw(image)
```

```python
# Собрать трек по видео
model = UltralyticsModel('yolo11n.pt', tracker='bytetrack.yaml')
df, file, true_frames = model.video2subtask('data/video.mp4')
```

## 🏷️ Разметка

### Разметить изображение

```python
import cv2
from ul_utils import UltralyticsModel

model = UltralyticsModel('yolo11n.pt', mode='preannotation')

image = cv2.imread('frame_01.jpg')[..., ::-1]
df = model.img2df(image)

print(df[['label', 'type', 'points']].head())
```

`img2df` возвращает датафрейм в схеме `cvat.new_df()`: `label`, `frame`,
`type`, `points`, `attributes` и остальные столбцы. Для детекции `type`
равен `rectangle`, для сегментации — `polygon` или `mask`.

### Показать превью

```python
model = UltralyticsModel('yolo11n-seg.pt', mode='preview')
preview = model.draw(image)
cv2.imwrite('preview.jpg', preview[..., ::-1])
```

В режиме `preview` вызов модели возвращает изображение с нарисованными
объектами, а не датафрейм.

## 🎯 Трекинг

### Трекинг в видео

```python
from ul_utils import UltralyticsModel

model = UltralyticsModel(
    'yolo11n.pt',
    tracker='bytetrack.yaml',
    mode='preannotation',
)

df, file, true_frames = model.video2subtask('data/video.mp4', desc='Трекинг')
```

`video2subtask` проходит видео потоком и собирает одну CVAT-подзадачу:
датафрейм, путь к файлу и словарь соответствия кадров. Трекер сбрасывается
перед началом.

## ⚙️ Постобработка

```python
from boxmot_utils import Tracker
from ul_utils import UltralyticsModel

model = UltralyticsModel(
    'yolo11n.pt',
    mode='preannotation',
    postprocess_filters=[Tracker(tracker_type='ocsort')],
)
```

Фильтры применяются по очереди к списку объектов. Фильтр может принимать
`(objs)` или `(objs, img)`; число аргументов определяется автоматически.

## ⚠️ Ограничения

- Модель ищется в `~/models`, если передано только имя файла и его нет рядом.
- `video2subtask` держит все датафреймы кадров в памяти до конца обработки.
- Трекинг доступен только для моделей Ultralytics с поддержкой трекера;
  путь к конфигу трекера задаётся строкой.
- Для масок выполняется обратное масштабирование к исходному размеру кадра,
  но паддинг снимается только для стандартного режима Ultralytics.
