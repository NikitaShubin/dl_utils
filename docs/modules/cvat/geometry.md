# 📐 cvat: геометрия контуров

`CVATPoints` — единый класс для прямоугольников, полигонов, эллипсов и
масок. Он хранит координаты, тип и поворот, умеет конвертировать форматы и
рисовать себя на изображении.

## 🚀 Примеры

```python
# Создать прямоугольник и полигон из YOLO-формата
from cvat import CVATPoints

rectangle = CVATPoints.from_yolobbox(0.5, 0.4, 0.2, 0.3, imsize=(1080, 1920))
polygon = CVATPoints.from_yoloseg(
    [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4],
    imsize=(1080, 1920),
)
```

```python
# Переместить полигон и обрезать его областью
from cvat import CVATPoints

polygon = CVATPoints.from_yoloseg(
    [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4],
    imsize=(1080, 1920),
)
shifted = polygon.shift([100, 50])
cropped = shifted.crop([100, 100, 500, 500])
```

```python
# Нарисовать контур на кадре
import cv2
from cvat import CVATPoints

image = cv2.imread('frame_01.jpg')[..., ::-1]
points = CVATPoints.from_yolobbox(0.5, 0.4, 0.2, 0.3, image.shape[:2])
image = points.draw(image, color=(255, 0, 0), thickness=2)
```

## 📐 Контуры

### Создать контур

```python
from cvat import CVATPoints

# Прямоугольник из YOLO-формата (cx, cy, w, h — доли кадра):
rectangle = CVATPoints.from_yolobbox(0.5, 0.4, 0.2, 0.3, imsize=(1080, 1920))

# Полигон из YOLO-сегментации:
polygon = CVATPoints.from_yoloseg(
    [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4],
    imsize=(1080, 1920),
)
```

Также доступны `CVATPoints(points, type_='polygon')` и
`CVATPoints.from_bbox(box, imsize, format='xyxy')`. По умолчанию `from_bbox`
ожидает формат `'xywh'`, поэтому формат нужно указывать явно.

### Преобразовать тип

```python
rectangle = polygon.asrectangle()
polygon = rectangle.aspolygon()
```

`asrectangle` вписывает контур в минимальный прямоугольник, `aspolygon`
превращает прямоугольник в четырёхточечный полигон.

### Переместить и масштабировать

```python
shifted = rectangle.shift([100, 50])
scaled = rectangle.scale([0.5, 0.5])
rotated = rectangle.rot90(1)
flipped = rectangle.fliplr(height=1080, width=1920)
```

Все методы возвращают новый объект. `shift` и `scale` принимают пару
`(по x, по y)`; `scale` пропорционально меняет и `imsize`. `rot90` принимает
число поворотов на 90 градусов, `flip` — ось (`0` или `1`).

### Обрезать

```python
cropped = polygon.crop([100, 100, 500, 500])
if cropped is None:
    print('Контур не пересекается с областью')
```

`crop` возвращает `None`, если пересечения нет.

### Площадь и центр

```python
print(polygon.size())     # площадь
print(polygon.center())   # (x, y)
```

Для самопересекающихся контуров площадь считается по формуле шнуровки и может
быть некорректной.

### Склеить многоконтурный объект

```python
fused = polygon.fuse_multipoly()
smart_fused = polygon.smart_fuse_multipoly()
```

CVAT иногда хранит объект с дырками как один контур с повторяющимися точками.
`fuse_multipoly` грубо убирает повторы, `smart_fuse_multipoly` пытается
сохранить форму.

### Упростить контур

```python
simplified = polygon.reducepoly(epsilon=2.0)
```

`epsilon` — максимальное отклонение в пикселях. Чем больше значение, тем
меньше точек останется.

### Конвертировать в объект

```python
from cv_utils import BBox, Mask

mask = Mask(**polygon.to_Mask_kwargs())
bbox = BBox(**rectangle.to_BBox_kwargs())
```

`to_Mask_kwargs` и `to_BBox_kwargs` возвращают словари для конструкторов
`cv_utils.Mask` и `cv_utils.BBox`.

### Нарисовать

```python
import cv2
from cvat import CVATPoints

image = cv2.imread('frame_01.jpg')[..., ::-1]
points = CVATPoints.from_yolobbox(0.5, 0.4, 0.2, 0.3, image.shape[:2])
image = points.draw(image, color=(255, 0, 0), thickness=2)
```

`draw` возвращает изображение; `show` выводит его через `matplotlib`.

### Треугольники

```python
from cvat import Triangle

triangles = Triangle.poly2triangles(polygon.points)
area = Triangle.triangles_area(triangles)
```

`Triangle` используется внутри `CVATPoints.size()` и `center()`. Вручную его
удобно применять для проверки площади сложных полигонов.

## ⚠️ Ограничения

- `CVATPoints` не хранит метку: она лежит в `attribs` и переносится в
  датафрейм через `to_dfrow`.
- Поворот применяется только к прямоугольникам и эллипсам; для полигонов
  `rotation` обычно равен нулю.
- `from_dfrow` без `imsize` не может перевести относительные координаты в
  абсолютные.
- `fuse_multipoly` эвристичен: на сложных дырках результат может отличаться от
  ожидаемого.
