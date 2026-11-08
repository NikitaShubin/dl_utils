# alb_utils.py — расширения Albumentations

Пылевая взвесь и удобная обёртка для совместных преобразований изображения и
маски.

**Зависимости:** `albumentations`, `numpy`, `utils.py`.

## Наложить пыль на изображение

```python
import cv2
from alb_utils import RandomDust

image = cv2.imread('frame_01.jpg')[..., ::-1]

dust = RandomDust(color=(210, 190, 160), fog_coef_range=(0.1, 0.3))
image_with_dust = dust(image=image)['image']
```

`RandomDust` наследуется от `A.RandomFog`, но не размывает картинку и
принимает цвет пыли. `color` задаётся в RGB, `fog_coef_range` — доля
запылённости.

## Преобразовать изображение и маску вместе

```python
import albumentations as A
from alb_utils import AlbTransforms

transform = AlbTransforms([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

image, mask = transform.image_mask_func(image, mask)
```

`image_mask_func(image, mask)` возвращает пару в том же порядке. Это удобно
передавать в `mpmap`, которая ожидает позиционные аргументы:

```python
from utils import mpmap

results = mpmap(transform.image_mask_func, images, masks, desc='Аугментация')
```

## Ограничения

- `AlbTransforms` не проверяет, что все преобразования поддерживают маску;
  `A.Compose` молча пропустит лишние цели.
- `RandomDust` требует трёхканальное изображение: цвет пыли раскладывается по
  каналам RGB.
- Параметры `fog_coef_range` и другие общие параметры относятся к
  `A.RandomFog`; их поведение меняется вместе с версией Albumentations.
