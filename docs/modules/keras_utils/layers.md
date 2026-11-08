# keras_utils: слои и блоки

Функции, собирающие фрагменты вычислительного графа Keras. Все они принимают
тензоры и возвращают тензоры, поэтому встраиваются в функциональную модель.

## Дать тензору имя

```python
from keras_utils import NamingLayer, layers, models

inp = layers.Input((256, 256, 3))
named = NamingLayer('features')(inp)
model = models.Model(inp, named)

print(model.get_layer('features'))
```

`NamingLayer` — `Lambda`-слой без вычислений. Имя пригодится для поиска узла
в графе через `get_model_elements`.

## Глобальный пулинг с bottleneck

```python
from keras_utils import global_pool_2D_with_bottleneck, layers

inp = layers.Input((64, 64, 128))
out = global_pool_2D_with_bottleneck(
    inp,
    bottleneck_size=0.25,
    out_size=1.0,
    mode='both',
)
```

Дробные `bottleneck_size` и `out_size` считаются как доли от числа каналов
входа. `mode` выбирает `avg`, `max` или `both`.

## Res-блок

```python
from keras_utils import layers, res_block

inp = layers.Input((64, 64, 64))
out = res_block(inp, lambda x: layers.Conv2D(64, 3, padding='same')(x))
```

`res_block` складывает вход и результат переданной функции. Число каналов и
размер должны совпадать, иначе `Add` бросит ошибку.

## Пирамидальный пулинг

```python
from keras_utils import layers, spatial_pyramid_pooling

inp = layers.Input((64, 64, 256))
out = spatial_pyramid_pooling(
    inp,
    dilation_rates=[0, 6, 12, 18],
    num_channels=256,
    dropout=0.1,
)
```

`dilation_rates=0` означает свёртку `1×1`; остальные значения — `3×3` с
прореживанием. Пустой список отключает пирамиду, оставляя только глобальный
пулинг.

## Глобальная карта признаков

```python
from keras_utils import global_pool_conv2D, layers

inp = layers.Input((64, 64, 256))
out = global_pool_conv2D(inp, mode='both', filters=512)
```

Функция строит вектор из `GlobalAvgPool` и `GlobalMaxPool`, прогоняет его
через `Conv2D + BN + ReLU` и растягивает обратно до размера входа.

## Масштабировать под образец

```python
from keras_utils import Resizing2DLike, layers

skip = layers.Input((128, 128, 64))
deep = layers.Input((32, 32, 256))

resized = Resizing2DLike(skip)(deep)
```

Слой приводит пространственные размеры `deep` к размерам `skip`, не меняя
число каналов. Требует полностью определённых размеров у образца.

## Узел фрактальной U-Net

```python
from keras_utils import fractal_unet_node, layers

skip = layers.Input((64, 64, 32))
down = layers.Input((16, 16, 64))

out, new_skip = fractal_unet_node(
    skip_connection=skip,
    donwsampled=down,
    out_filters=32,
    latent_filters=16,
    use_out_skip=True,
)
```

Узел объединяет до трёх входов (`skip_connection`, `upsampled`,
`donwsampled`), применяет `1×1 → DepthwiseConv → 1×1` и возвращает результат
вместе с новым skip-тензором. Имя параметра `donwsampled` сохранено как в
исходном коде.

## Универсальный вход

```python
from keras_utils import InputModel

img_input, inputs = InputModel(
    input_tensor=None,
    input_shape=(512, 512, 3),
    input_batch_size=None,
)
```

Если `input_tensor` задан, функция возвращает его и исходный вход графа —
удобно, когда модель встраивается в чужую предобработку.

## Ограничения

- Все функции рассчитаны на функциональный API Keras; в `Sequential` их
  можно использовать только как слои.
- `global_pool_2D_with_bottleneck` и `Resizing2DLike` требуют определённого
  числа каналов или размера; при `None` в shape будет ошибка.
- `res_block` не подгоняет число каналов: несовпадение размеров — ошибка
  сборки.
- `fractal_unet_node` не проверяет согласованность размеров входов.
- `global_pool_conv2D` при `filters=0` возвращает `None`; вызывающий код
  должен это учитывать.
