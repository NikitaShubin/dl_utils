# 🎬 utils: изображения и видео

Бытовые операции с кадрами: прочитать нужный кадр, вписать в размер, собрать
превью, наложить подпись или водяной знак.

## 🚀 Примеры

```python
from utils import ImReadBuffer

# Прочитать кадр из видео и из папки изображений
with ImReadBuffer() as frames:
    first = frames('video.mp4', frame=0)
    frame = frames('frames_dir', frame=5)
```

```python
import cv2
from utils import resize_with_pad

# Вписать кадр в квадрат с полями
img = cv2.imread('frame.jpg')
square = resize_with_pad(img, (512, 512), padding_color=(0, 0, 0))
```

```python
from utils import img_dir2video

# Собрать видео из папки кадров
img_dir2video('frames', video_file='preview.mp4', fps=5, desc='Сборка превью')
```

```python
from utils import draw_contrast_text, overlap_with_alpha

# Наложить подпись и водяной знак
img = draw_contrast_text(img, 'Кадр 42\nПроверка')
img = overlap_with_alpha(img, watermark_rgba)
```

## 📖 Чтение

### Прочитать кадр из видео или папки

`ImReadBuffer` держит файл открытым и умеет читать как видео, так и набор
изображений:

```python
from utils import ImReadBuffer

with ImReadBuffer() as frames:
    first = frames('video.mp4', frame=0)
    tenth = frames('video.mp4', frame=10)
    frame = frames('frames_dir', frame=5)  # папка с изображениями
```

Кадры возвращаются в RGB. Для изображений `frame` — индекс в отсортированном
списке файлов.

## ↔️ Изображение

### Вписать изображение в размер

`resize_with_pad` сохраняет пропорции и добавляет поля:

```python
import cv2
from utils import resize_with_pad

img = cv2.imread('frame.jpg')
square = resize_with_pad(img, (512, 512), padding_color=(0, 0, 0))
```

### Обрезка полей и плёнка

```python
from utils import autocrop, Img2Film

cropped = autocrop(img)          # убирает однотонные края
film = Img2Film()(cropped)       # обрамляет кадром фотоплёнки
```

`autocrop` сравнивает верхний и левый края с цветом левого верхнего пикселя,
а нижний и правый — с цветом правого нижнего. Если изображение однотонное,
функция возвращает `None`. `Img2Film` удобен для превью и демонстраций, но
добавляет поля и меняет размер изображения.

## 🧱 Видео

### Собрать видео из папки

```python
from utils import img_dir2video

img_dir2video(
    'frames',
    video_file='preview.mp4',
    fps=5,
    desc='Сборка превью',
)
```

Функция сама масштабирует кадры разных размеров и по умолчанию пересжимает
результат в H.264.

### Параметры видео

```python
from utils import get_video_info

info = get_video_info('video.mp4')
print(info['width'], info['height'], info['length'], info['fps'])
```

## 🏷️ Подписи

### Подписи и водяные знаки

```python
from utils import draw_contrast_text, overlap_with_alpha

img = draw_contrast_text(img, 'Кадр 42\nПроверка')
img = overlap_with_alpha(img, watermark_rgba)
```

`draw_contrast_text` рисует белый текст с тёмной обводкой в левом верхнем углу.
`overlap_with_alpha` ждёт водяной знак с альфа-каналом (`RGBA` или `LA`),
совпадающий с изображением по высоте и ширине.

## ⚠️ Ограничения

- `get_video_info` доверяет заголовку контейнера: для битых файлов число
  кадров может быть неточным.
- `ImReadBuffer` не потокобезопасен; для параллельного чтения создавайте
  отдельный экземпляр на процесс или поток.
- `img_dir2video` вызывает `ffmpeg` для межкадрового сжатия; без `ffmpeg`
  используйте `intra_frame_compression_only=True` и расширение `.avi`.
