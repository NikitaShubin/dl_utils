# keras_utils: модели

Готовые сегментационные архитектуры и обвязка вокруг предобученных
backbone-ов из `keras.applications`.

Данные и загрузчик описаны в разделе [«Данные»](data.md), метрики и
колбеки — в [«Обучение»](training.md).

## Рассчитать размеры backbone-а

```python
from keras_utils import BackboneShapes

shapes = BackboneShapes(
    true_input_shape=(1200, 1600),
    desirable_scale=1 / 1.6,
    num_poolings=5,
)

print(shapes.input_shape)   # [736, 992]
print(shapes.output_shape)  # [23, 31]
```

`BackboneShapes` подбирает входной размер так, чтобы после всех пулингов
остались целые размеры, и считает размер выходной карты признаков.

## Взять backbone с предобработкой

```python
from keras_utils import backbone_with_preprop

backbone = backbone_with_preprop(
    'MobileNetV3Large',
    input_shape=(512, 512, 3),
    trainable=False,
)
```

Функция создаёт модель без головы, добавляет `Rescaling` в диапазон `[-1, 1]`
и по умолчанию замораживает веса. Предобученные веса берутся из
`imagenet`; при отсутствии интернета конструктор упадёт.

## Получить многомасштабные признаки

```python
import numpy as np

from keras_utils import backbone2encoder, backbone_with_preprop

backbone = backbone_with_preprop('MobileNetV3Large', input_shape=(512, 512, 3))
encoder = backbone2encoder(backbone)

features = encoder.predict(np.zeros((1, 512, 512, 3)))
print([f.shape for f in features])
```

`backbone2encoder` оставляет выходы слоёв, размер которых уменьшается ровно в
два раза. Это входы для skip-соединений U-Net-подобных сетей.

## Собрать U-Net

```python
from keras_utils import UNet

model = UNet(
    backbone='MobileNetV3Large',
    input_shape=(512, 512, 3),
    out_filters=4,
    feat_filters=16,
    pool_filters=256,
    dropout_rate=0.1,
)
model.summary()
```

`use_submodels=True` инкапсулирует backbone и голову в подмодели — удобно для
квантизации. `activation='argmax'` возвращает номера классов вместо
вероятностей; используйте это только на инференсе.

`out_filters` — число классов, включая фон. В многоклассовом режиме оно должно
совпадать с `num_classes` в [`SegDataLoader`](data.md), иначе форма one-hot
маски не сойдётся с выходом модели. При `out_filters > 2` включается
`softmax`, при `out_filters <= 2` — `sigmoid`; для бинарной сегментации берите
`out_filters=1` и не задавайте `num_classes`. `out_filters=2` даёт два
независимых `sigmoid`-канала, а не `softmax`-распределение, поэтому для пары
«фон + объект» обычно используют бинарный режим.

`input_shape` задаётся в порядке Keras: `(высота, ширина, каналы)`. Например,
`(512, 512, 3)` — RGB-изображение 512×512. `SegDataLoader` уже отдаёт
`float32` в `[0, 1]`, поэтому делить на 255 не нужно. `BackboneShapes` при этом
принимает `true_input_shape=(высота, ширина)` и возвращает `input_shape` в том
же порядке `[высота, ширина]`.

`UNet` собирает backbone через `backbone_with_preprop` с `trainable=False`, то
есть по умолчанию веса заморожены. Чтобы дообучить backbone, разморозьте
нужные слои и вызовите `recompile_model` — иначе Keras продолжит использовать
старый граф обучения. Пример есть в разделе [«Обучение»](training.md).

## Собрать Deeplabv3+

```python
from keras_utils import Deeplabv3Plus

model = Deeplabv3Plus(
    backbone='MobileNetV3Large',
    input_shape=(512, 512, 3),
    out_filters=4,
    en_lvl_4_dec=2,
    spp_dilations=[0, 6, 12, 18],
)
```

`en_lvl_4_dec` задаёт уровень низкоуровневой карты признаков для обходного
пути. Отрицательное значение считается от конца списка признаков; `None`
отключает обходной путь.

## Заменить голову

```python
from keras_utils import replace_head

model = replace_head(model, filters=10, activation='softmax')
```

Функция находит последний `Conv2D`, меняет число фильтров и активацию, а
остальные слои замораживает. Если число фильтров совпадает, слой не
пересобирается — меняется только активация.

## Найти слои по имени

```python
from keras_utils import get_all_layers, get_model_elements, layers

convs = get_model_elements(model, element_class=layers.Conv2D)
print(list(convs)[:5])

all_layers = get_all_layers(model)
```

`get_model_elements` возвращает словарь `путь_индексов → слой` и умеет
искать по части имени. `get_all_layers` разворачивает подмодели в плоский
список.

## Фрактальная U-Net

```python
from keras_utils import make_fractal_unet

model = make_fractal_unet('MobileNetV3Large', im_size=(256, 256))
```

Экспериментальная архитектура с несколькими проходами по skip-соединениям.
Требует backbone ровно с шестью уровнями признаков и всегда выдаёт три
класса.

## Ограничения

- `backbone2encoder` при неопределённом входе прогоняет нулевой тензор через
  модель; это замедляет сборку и требует памяти.
- `backbone_with_preprop` добавляет `Rescaling(2, -1)`, то есть ожидает вход
  в диапазоне `[0, 1]`, а не `[0, 255]`.
- `UNet` не проверяет, что размер входа делится на число пулингов; ошибка
  проявится при сборке skip-соединений.
- `Deeplabv3Plus` при передаче готовой модели требует не меньше пяти выходов:
  если передать обычный backbone, сборка упадёт с `ValueError`. При передаче
  имени энкодер строится автоматически.
- `make_fractal_unet` жёстко задаёт три выходных класса и число уровней;
  аргумент `out_filters` влияет только на промежуточные слои.
- `replace_head` ищет только `Conv2D`; модели с другой головой он не
  адаптирует.
