# Модуль [`boxmot_utils.py`](../boxmot_utils.py)

Содержит обёртки для работы с трекерами библиотеки BoxMOT, обеспечивающие удобный
интерфейс для трекинга объектов для работы с CVAT).

Основной фокус сделан на многокадровом отслеживании объектов с поддержкой
различных алгоритмов трекинга (OC-SORT, BoT-SORT, ByteTrack и др.).

---

## Основные компоненты

### Классы

| Класс     | Описание                                              |
|-----------|-------------------------------------------------------|
| `Tracker` | Обёртка для трекеров BoxMOT для работы с CVAT-данными |

---

## Класс `Tracker`

Обёртка вокруг трекера из BoxMOT для работы с CVAT-данными.

### Конструктор

```python
Tracker(
    tracker_type: str = 'ocsort',
    *,
    store_untracked: bool = False,
    **tracker_kwargs
)
```

#### Параметры
<!-- markdownlint-disable MD013 -->
| Параметр           | Тип    | По умолчанию | Описание                                       |
|--------------------|--------|--------------|------------------------------------------------|
| `tracker_type`     | `str`  | `'ocsort'`   | Тип трекера (см. список ниже)                  |
| `store_untracked`  | `bool` | `False`      | Возвращать все объекты или только с `track_id` |
| `**tracker_kwargs` | `dict` | `{}`         | Дополнительные параметры трекера               |
<!-- markdownlint-enable MD013 -->
##### Доступные типы трекеров

- `strongsort` - StrongSORT с ReID
- `ocsort` - Observation-Centric SORT
- `bytetrack` - ByteTrack
- `botsort` - BoT-SORT с ReID
- `deepocsort` - Deep OC-SORT с ReID
- `hybridsort` - Hybrid SORT с ReID
- `boosttrack` - BoostTrack с ReID

##### Параметры ReID (для трекеров с `*`)

Для трекеров: `strongsort`, `botsort`, `deepocsort`, `hybridsort`, `boosttrack`
<!-- markdownlint-disable MD013 -->
| Параметр       | Тип          | По умолчанию                     | Описание                               |
|----------------|--------------|----------------------------------|----------------------------------------|
| `reid_weights` | `Path`       | `~/models/osnet_x0_25_msmt17.pt` | Путь к весам модели ReID               |
| `device`       | `str`/`None` | `None`                           | Устройство: `'cuda'`, `'cpu'` или авто |
| `half`         | `bool`       | `False`                          | Использовать float16                   |
<!-- markdownlint-enable MD013 -->
##### Общие параметры трекеров

| Параметр        | Тип     | Описание                                 |
|-----------------|---------|------------------------------------------|
| `det_thresh`    | `float` | Порог детекции                           |
| `max_age`       | `int`   | Макс. возраст трека (кадры)              |
| `max_obs`       | `int`   | Макс. число наблюдений на трек           |
| `min_hits`      | `int`   | Мин. хитов для подтверждения трека       |
| `iou_threshold` | `float` | Порог IoU для ассоциации                 |
| `per_class`     | `bool`  | Раздельное отслеживание по классам       |
| `nr_classes`    | `int`   | Число классов (при `per_class=True`)     |
| `asso_func`     | `str`   | Алгоритм ассоциации                      |
| `is_obb`        | `bool`  | Использовать OBB (ориентированные боксы) |

### Методы

#### `__call__()`

```python
__call__(objs, img) -> list[BBox | Mask]
```

Применяет трекер к очередному кадру и его объектам.

**Параметры:**

- `objs` - Список объектов детекции (BBox или Mask)
- `img` - Изображение кадра в формате RGB

**Возвращает:**

- Список объектов с атрибутом `track_id`

#### `reset()`

```python
reset() -> None
```

Сбрасывает внутренние состояния трекера.

### Примеры использования

#### Базовое использование

```python
# Трекинг объектов, детектируемых Grounding DINO:
img2df = GDino(
    ...,
    postprocess_filters=[NMS(), Tracker()]
)
```

```python
tracker = Tracker(tracker_type='botsort')
for frame in VideoGenerator('./test.avi'):
    # Нечто, возвращающее список обнаруженных на изображении
    # объекты в виде BBox или Mask из cv_utils:
    objects = img2objs(frame)

    tracked_objects = tracker(objects, frame)
    # Каждому объекту добавлен атрибут 'track_id', а объекты без трека исключены.
```

#### Все объекты

```python
tracker = Tracker(tracker_type='bytetrack', store_untracked=True)
```

#### Трекер с ReID и кастомными параметрами

```python
tracker = Tracker(
    tracker_type='strongsort',
    reid_weights='path/to/custom_model.pt',
    device='cuda',
    half=True,
    det_thresh=0.4,
    max_age=30
)
```

#### ByteTrack с порогами ассоциации

```python
tracker = Tracker(
    tracker_type='bytetrack',
    track_thresh=0.6,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=30
)
```

#### OC-SORT с адаптивным порогом

```python
tracker = Tracker(
    tracker_type='ocsort',
    det_thresh=0.3,
    min_conf=0.1,
    use_byte=True,
    inertia=0.2
)
```

---

## Примечания

### Формат данных

- Объекты должны иметь атрибуты `attribs['confidence']` (float) и
   `attribs['label']` (str)
- Изображения должны быть в формате RGB

### Работа с треками

- Треки нумеруются с 0
- При смене сцены вызывайте `reset()` для очистки истории

### Производительность

- Для трекеров с ReID рекомендуется GPU (`device='cuda'`)
- Полу-точность (`half=True`) ускоряет вычисления на современных GPU

### Совместимость

- Поддерживает как Bounding Boxes (`BBox`), так и маски (`Mask`) из cv_utils.py
- Совместим с форматом аннотаций CVAT из cvat.py
