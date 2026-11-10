# 🔍 gdinoal.py — Grounding DINO

Открытая детекция по текстовым запросам: модель находит объекты по описанию и
сразу переводит их в CVAT-датафрейм.

**Зависимости:** `groundingdino`, `torch`, `opencv-python`, `matplotlib`,
`pt_utils.py`, `cvat.py`, `cv_utils.py`, `ml_utils.py`, `utils.py`.

## 🚀 Примеры

```python
# Найти объекты по описанию
from gdinoal import GDino

model = GDino(prompt2label={'Person': 'person', 'Car': 'car'})
df = model.img2df(image)
```

```python
# Нарисовать результат поверх кадра
preview = model.draw(image)
```

```python
# Добавить NMS и трекинг к результатам
from boxmot_utils import Tracker
from cv_utils import NMS
from gdinoal import GDino

model = GDino(
    prompt2label={'Person': 'person'},
    postprocess_filters=[NMS(), Tracker(tracker_type='ocsort')],
)
```

```python
# Задать запрос строкой
model = GDino(prompt2label='Person')
df = model(image)
```

## 🔍 Детекция

### Найти объекты по описанию

```python
import cv2
from gdinoal import GDino

model = GDino(
    box_threshold=0.3,
    text_threshold=0.3,
    prompt2label={
        'Person': 'person',
        'Car': 'car',
        'Traffic light': 'traffic_light',
    },
)

image = cv2.imread('frame_01.jpg')[..., ::-1]
df = model.img2df(image)

print(df[['label', 'points', 'attributes']])
```

Ключи `prompt2label` — текстовые запросы для модели, значения — метки в вашей
разметке. Один запрос может отображаться в несколько формулировок, и наоборот,
несколько запросов могут давать одну метку.

### Один запрос строкой

```python
model = GDino(prompt2label='Person')
df = model(image)  # метка будет равна самому запросу
```

Если `prompt2label` — строка, она используется и как запрос, и как метка.

### Пакетная обработка кадров

```python
for frame in frames:
    df = model.img2df(frame, df_as_list_of_dfs=True)
    # df — список однострочных датафреймов, по одному на объект
```

`df_as_list_of_dfs=True` полезен, когда объекты нужно обрабатывать по одному.

## 🎨 Визуализация

### Нарисовать результат

```python
preview = model.draw(image)
cv2.imwrite('preview.jpg', preview[..., ::-1])

# Или показать в ноутбуке:
model.show(image)
```

## ⚙️ Постобработка

```python
from boxmot_utils import Tracker
from cv_utils import NMS
from gdinoal import GDino

model = GDino(
    prompt2label={'Person': 'person'},
    postprocess_filters=[NMS(), Tracker(tracker_type='ocsort')],
)
```

Фильтры применяются к списку `BBox`; фильтр может принимать `(objs)` или
`(objs, img)`. `NMS` убирает дубли, `Tracker` добавляет `track_id`.

## ⚠️ Ограничения

- Модель скачивается в `~/models`, если передано только имя файла.
  Поддерживаются `groundingdino_swint_ogc.pth` и
  `groundingdino_swinb_cogcoor.pth`; другое имя заменяется на вторую модель.
- Порог `box_threshold` отсекает рамки, `text_threshold` — совпадение текста.
  Слишком низкие значения дают много ложных срабатываний.
- Модель не обучается на ваших классах: качество зависит от формулировки
  запросов.
- `img2df` возвращает пустой датафрейм, если ничего не найдено.
