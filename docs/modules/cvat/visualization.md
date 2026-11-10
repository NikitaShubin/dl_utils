# 🎨 cvat: визуализация

Отрисовка разметки на кадре и сборка превью-видео для подзадачи или списка
задач.

## 🚀 Примеры

```python
# Нарисовать разметку кадра с полупрозрачной заливкой
import cv2
from cvat import ergonomic_draw_df_frame
from utils import get_n_colors

image = cv2.imread('frame_01.jpg')[..., ::-1]
labels = sorted(df['label'].unique())
label2color = dict(zip(labels, get_n_colors(len(labels))))
frame_df = df[df['frame'] == 0]
preview = ergonomic_draw_df_frame(frame_df, image, label2color=label2color)
```

```python
# Нарисовать все сегменты на пустом холсте
from cvat import df2img

canvas = df2img(df, label2color, imsize=(1080, 1920))
```

```python
# Собрать превью-видео подзадачи
from cvat import subtask2preview

subtask2preview(
    subtask,
    out_file='results/preview.mp4',
    label2color=label2color,
    fps=3,
)
```

## 🎨 Рисование

### Нарисовать кадр

```python
import cv2
from cvat import ergonomic_draw_df_frame
from utils import get_n_colors

image = cv2.imread('frame_01.jpg')[..., ::-1]

labels = sorted(df['label'].unique())
label2color = dict(zip(labels, get_n_colors(len(labels))))

frame_df = df[df['frame'] == 0]
preview = ergonomic_draw_df_frame(
    frame_df,
    image,
    label2color=label2color,
    alpha=0.5,
)
cv2.imwrite('preview.jpg', preview[..., ::-1])
```

`ergonomic_draw_df_frame` заливает сегменты полупрозрачным цветом и обводит
контуры. Для строгой отрисовки без заливки используйте `draw_df_frame` с
`thickness`.

### Нарисовать все сегменты на пустом холсте

```python
from cvat import df2img

canvas = df2img(df, label2color, imsize=(1080, 1920))
```

Если передать `img`, разметка ляжет поверх него.

## 👁️ Превью

### Превью подзадачи

```python
from cvat import subtask2preview

subtask2preview(
    subtask,
    out_file='results/preview.mp4',
    label2color=label2color,
    alpha=0.5,
    fps=3,
    desc='Превью',
)
```

Функция интерполирует разметку, читает исходное видео и пишет ролик, накладывая
цветные сегменты и номер кадра. Для `.mp4` автоматически выполняется
пересжатие.

### Превью списка задач

```python
from cvat import tasks2preview

tasks2preview(
    tasks,
    out_file='results/all_preview.mp4',
    label2color=label2color,
    fps=3,
    desc='Общее превью',
)
```

Каждая задача пишется в отдельный файл `all_preview_00000000.mp4`, затем они
склеиваются в один.

## ⚙️ Постобработка

### Постобработка кадров

```python
from video_utils import AddCaption, Resize

subtask2preview(
    subtask,
    out_file='results/preview.mp4',
    postprocessor=Resize(720),
)
```

`postprocessor` — функция или конвейер, применяемый к каждому кадру перед
записью.

## ⚠️ Ограничения

- `df2img` требует `label2color` для каждой метки; отсутствующий ключ даст
  `KeyError`.
- `subtask2preview` читает всё видео и хранит его генератор; для длинных
  записей это занимает время.
- `tasks2preview` не проверяет код возврата `ffmpeg`: при ошибке склейки
  временные файлы всё равно удаляются.
- `ergonomic_draw_df_frame` работает с RGB; не передавайте BGR без
  преобразования.
