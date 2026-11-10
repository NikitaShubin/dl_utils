# 🪄 samal.py — автоматическая сегментация SAM

Segment Anything Model HQ для разметки изображений: модель находит все объекты
и возвращает готовую CVAT-подзадачу.

**Зависимости:** `segment-anything-hq`, `torch`, `numpy`, `opencv-python`,
`pandas`, `tqdm`, `cvat.py`, `cv_utils.py`, `pt_utils.py`, `ml_utils.py`,
`utils.py`.

## 🏷️ Разметить изображение

```python
import cv2
from samal import SAM

sam = SAM(model_path='sam_hq_vit_h.pth')
image = cv2.imread('frame_01.jpg')[..., ::-1]

df = sam.img2df(image)
print(df[['label', 'points', 'attributes']])
```

`SAM` загружает одну или несколько моделей и объединяет их маски. По умолчанию
включён TTA: изображение отражается и поворачивается, результаты совмещаются.
Это точнее, но примерно в 8 раз медленнее.

## 🧱 Собрать CVAT-подзадачу из файла

```python
subtask = sam.img_file2subtask('frame_01.jpg')
df, file, true_frames = subtask
```

`img_file2subtask` читает файл, применяет модель и возвращает кортеж
`(df, file, true_frames)` — ровно то, что ждут функции `cvat.py`.

## ⚙️ Постобработка масок

```python
from samal import SAM
from seg import DropByArea, Morphology

sam = SAM(
    model_path='sam_hq_vit_h.pth',
    postprocess_filters=[
        DropByArea(min_area=0.001),
        Morphology('close', 5),
    ],
)
```

Фильтры получают список `Mask` и возвращают новый список. Фильтр может
принимать `(masks)` или `(masks, img)`.

## 🧩 Несколько моделей

```python
sam = SAM(model_path=['sam_hq_vit_h.pth', 'sam_hq_vit_l.pth'])
```

Маски всех моделей складываются в один список. Имена файлов должны содержать
тип модели (`vit_h`, `vit_l`, `vit_b`, `vit_tiny`).

## ♻️ Сбросить состояние

```python
sam.reset()
```

## ⚠️ Ограничения

- Модели скачиваются в `~/models`, если передано только имя файла.
- `SAM` не различает классы: в датафрейме заполняется `source='SAM'`, а
  `label` остаётся пустым (`None`). Метку можно задать аргументом `label` в
  `task_auto_annottation` или через `LabelsConvertor`.
- TTA улучшает контуры, но требует много памяти и времени; для видео обычно
  отключайте его (`use_tta=False`).
- `masks2df` векторизует маски в полигоны; для сложных объектов контур может
  содержать тысячи точек.
- Модуль предназначен для изображений; для видео используйте `sam2al.py`.
