# 🎞️ video_utils.py — видео, фильтры и оптический поток

Чтение и запись видео, конвейеры покадровой обработки, готовые фильтры,
перепаковка битых файлов и геометрические искажения по оптическому потоку.

**Зависимости:** `opencv-python`, `numpy`, `scipy`, `scikit-image`, `av`,
`tqdm`, `utils.py`.

## 🚀 Примеры

```python
# Прочитать видео и записать результат
from video_utils import ViRead, ViSave

with ViRead('data/video.mp4') as reader, ViSave('out.mp4', fps=30) as writer:
    for frame in reader:
        writer(frame)
```

```python
# Собрать конвейер фильтров
from video_utils import AddCaption, Pipeline, Resize

pipeline = Pipeline([Resize(512), AddCaption('Project 1 / Task 1')])
pipeline.convert('data/video.mp4', 'results/preview.mp4', step=2)
```

```python
# Восстановить кадр по оптическому потоку
from video_utils import OptFlow

flow = OptFlow()(frame_1, frame_2)
restored = OptFlow.apply_flow2img(flow, frame_1)
```

```python
# Применить одно искажение к кадру и маске
from video_utils import ImDistort

distort = ImDistort.from_example(frame_1, frame_2)
warped = distort(frame_1)
mask_warped = distort(mask, mask_mode=True)
```

## 📖 Чтение

### Прочитать видео

```python
from video_utils import ViRead

with ViRead('data/video.mp4', colorspace='rgb') as frames:
    for frame in frames:
        print(frame.shape)  # (height, width, 3), RGB
```

`ViRead` умеет начинать с заданного кадра (`start_frame=100`) или со
случайного кадра из диапазона (`start_frame=(100, 200)`). Отрицательные
значения считаются от конца. `on_end` задаёт поведение на конце потока:
`'close'` (по умолчанию), `'reset'` или `'repeat_last_frame'`.

## 💾 Запись

### Записать видео

```python
from video_utils import ViRead, ViSave

with ViRead('data/video.mp4') as reader, ViSave('results/out.mp4', fps=30) as writer:
    for frame in reader:
        writer(frame)
```

`ViSave` принимает RGB по умолчанию и сам переводит кадры в BGR. Папка
назначения создаётся автоматически.

### Перепаковать битое видео

```python
from video_utils import clean_remux

clean_remux('data/broken.mp4', 'results/fixed.mp4')
```

Функция копирует пакеты без пересжатия и выбрасывает те, которые декодер не
может превратить в кадр.

### Пересжать в mp4

```python
from video_utils import recomp2mp4

recomp2mp4('results/tmp.avi', 'results/final.mp4')
```

## 🧱 Фильтры

### Собрать конвейер фильтров

```python
from video_utils import Pipeline, Resize, AddCaption

pipeline = Pipeline([
    Resize(512),
    AddCaption('Project 1 / Task 1'),
])

pipeline.convert('data/video.mp4', 'results/preview.mp4', step=2)
```

`step=2` обрабатывает каждый второй кадр. Строки в списке — синонимы готовых
фильтров, числа — пороги бинаризации. Вложенные списки разворачиваются.

### Применить фильтр к папке

```python
from video_utils import apply2video

apply2video(Resize(512), 'data/videos', 'results/videos', step=1)
```

Если `inp_file` — папка, обрабатываются все видео в ней. `skip_existed=True`
пропускает уже готовые файлы. Для изображений есть `apply2image`.

### Готовые фильтры

```python
from video_utils import (
    AddCaption, AsType, Concat, Mix, PutTime, Res,
    Resize, StoreLastFrames,
)

pipeline = Pipeline([
    Resize((720, 1280)),
    StoreLastFrames(3),      # возвращает список из 3 кадров
    Mix(Resize(360), 0.4),   # накладывает уменьшенную версию
])
```

`Res` и `Concat` собирают коллажи, `PutTime` пишет время или номер кадра,
`CompareTwoFilters` и `CompareFiltersWithTarget` помогают сравнивать фильтры.

## 🌊 Оптический поток

```python
from video_utils import OptFlow

flow = OptFlow()(frame_1, frame_2)
restored = OptFlow.apply_flow2img(flow, frame_1)
```

`OptFlow` оборачивает `cv2.calcOpticalFlowFarneback`. Последовательность можно
обработать целиком:

```python
flows = OptFlow().seq_flows(frames)
restored = OptFlow().apply_flows2img(flows, frames[0])
```

## 🎲 Повторяемое искажение

```python
from video_utils import ImDistort

distort = ImDistort.from_example(frame_1, frame_2)
warped = distort(frame_1)
mask_warped = distort(mask, mask_mode=True)
```

`ImDistort` хранит координатную сетку, поэтому одно и то же искажение можно
применять к изображениям и маскам. `from_opt_flow` строит искажение по уже
вычисленному потоку.

## ⚠️ Ограничения

- `ViRead` читает кадры BGR из OpenCV и конвертирует в выбранную схему;
  `colorspace='bgr'` оставляет как есть.
- `count_frames()` делает отдельный проход по файлу и может быть медленным.
- `Pipeline.convert` при пересжатии создаёт временный `*_tmp.avi`; если такой
  файл уже есть, будет `FileExistsError`.
- `ImDistort` использует интерполяцию высокого порядка; для масок передавайте
  `mask_mode=True`, иначе появятся промежуточные значения.
- `apply2video` по умолчанию берёт абсолютные пути автора, если не передан
  `inp_file`; всегда указывайте источник явно.
