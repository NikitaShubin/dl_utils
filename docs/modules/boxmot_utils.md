# 🎯 boxmot_utils.py — трекинг объектов

Обёртка над трекерами boxmot: принимает объекты `BBox` и `Mask`, возвращает их
же с заполненным `attribs['track_id']`.

**Зависимости:** `boxmot`, `cv_utils.py`, `numpy`, `opencv-python`, `loguru`.

## 🎯 Отследить объекты в видео

```python
import cv2
from boxmot_utils import Tracker
from cv_utils import BBox

tracker = Tracker(tracker_type='ocsort')

cap = cv2.VideoCapture('data/video.mp4')
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # Детекции текущего кадра; например, из YOLO:
    objects = [
        BBox.from_yolo([0.5, 0.4, 0.2, 0.3], imsize=frame.shape[:2],
                       attribs={'label': 'car', 'confidence': 0.87}),
    ]

    tracked = tracker.update(objects, frame)
    for obj in tracked:
        print(obj.attribs['track_id'], obj.attribs['label'])
```

Трекер работает по кадрам: вызывайте `update` для каждого кадра подряд, начиная
с первого. Изображение передаётся в RGB.

## 📋 Что нужно объекту

У каждого объекта должны быть атрибуты:

- `attribs['label']` — строка, имя класса;
- `attribs['confidence']` — уверенность детектора;
- геометрия — `xyxy` для `BBox` или маска для `Mask`.

`Tracker` сам переводит метки в числовые классы и запоминает соответствие.
Повторно трекать объект с уже заполненным `track_id` нельзя — будет
`AttributeError`.

## 🚫 Объекты без трека

По умолчанию возвращаются только объекты, которым назначен трек. Чтобы
получить весь список:

```python
tracker = Tracker(tracker_type='ocsort', store_untracked=True)
objects = tracker.update(objects, frame)
for obj in objects:
    print(obj.attribs['track_id'])  # int или None
```

## 🎯 Трекеры с ReID

Трекерам вроде `botsort` нужны веса ReID-модели. По умолчанию ожидается
`~/models/osnet_x0_25_msmt17.pt`; путь и устройство можно переопределить:

```python
tracker = Tracker(
    tracker_type='botsort',
    reid_weights='~/models/osnet_x0_25_msmt17.pt',
    device='cuda',
    half=True,
)
```

## ♻️ Сбросить трекер

```python
tracker.reset()
```

Создаётся новый экземпляр трекера: нумерация начинается заново, накопленные
треки теряются.

## ⚠️ Ограничения

- Нумерация `track_id` начинается с нуля, хотя boxmot возвращает номера с
  единицы.
- Изображение конвертируется из RGB в BGR внутри `update`; не передавайте BGR.
- `store_untracked=True` возвращает и объекты без трека, поэтому фильтровать
  их придётся самостоятельно.
- `suppress_module_logs()` прячет логи boxmot только на время контекста.
