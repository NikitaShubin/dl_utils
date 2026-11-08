# tf_utils.py — утилиты TensorFlow

Инициализация TensorFlow, работа с несколькими GPU и готовые `tf.data`
конвейеры для сегментационных датасетов.

**Зависимости:** `tensorflow`, `alb_utils.py`, `utils.py`.

## Инициализировать TensorFlow

```python
from tf_utils import TFInit

TFInit(less_gpu_mem=True, log_level='ERROR')
```

`TFInit` можно использовать и как контекстный менеджер — тогда уровень
логирования вернётся к исходному после выхода:

```python
from tf_utils import TFInit

with TFInit(less_gpu_mem=True):
    model.fit(train_ds, epochs=10)
```

`less_gpu_mem=True` включает рост памяти GPU по мере необходимости вместо
захвата всей памяти сразу.

## Учесть несколько GPU

```python
from tf_utils import MultiGPU

gpus = MultiGPU()
print(gpus.num, gpus.multi, gpus.any)

batch_size = gpus.batch_size_multiplier(8)
with gpus.strategy.scope():
    model = build_model()
```

`batch_size_multiplier` умножает размер батча на число GPU. Если GPU нет,
целое значение `default` задаёт новый размер, а дробное — коэффициент.

## Загрузить датасет из папок

Ожидается структура:

```text
dataset/
    train/
        inp/   # изображения
        out/   # маски
    val/
    test/
```

```python
from tf_utils import get_datasets

train_ds, val_ds, test_ds = get_datasets(
    'dataset',
    num_classes=4,
    im_size=(608, 800),
    batch_size=8,
)
```

Изображения читаются из `inp`, маски — из `out`. Маски переводятся в one-hot
по `num_classes`. `num_classes` — число каналов, включая фон; значения маски
должны укладываться в `0..num_classes-1`. Пары файлов сопоставляются по
порядку сортировки, поэтому имена входов и выходов должны совпадать.

`im_size` задаётся как `(высота, ширина)` и должен совпадать с
`UNet.input_shape[:2]`. Изображение ресайзится `bilinear`, маска — `nearest`,
чтобы не появились дробные номера классов. Аугментации получают изображение в
`uint8` до деления на 255; в `SegDataLoader` порядок другой — сначала
нормализация в `[0, 1]`, потом преобразования.

Один датасет без разбиения:

```python
from tf_utils import get_dataset

ds = get_dataset(
    'dataset/train',
    num_classes=4,
    target_size=(608, 800),
    batch_size=8,
    shuffle=True,
)
```

## Ограничения

- `get_dataset` ориентирован на сегментацию: маска обязана быть одноканальным
  изображением, иначе `tf.one_hot` возьмёт только первый канал.
- `get_datasets` всегда делает one-hot по `num_classes`, поэтому бинарный режим
  `UNet(out_filters=1)` с ним не работает. Для бинарной сегментации берите
  [`SegDataLoader`](keras_utils/data.md).
- При `shuffle=True` последний неполный батч отбрасывается, чтобы не ломать
  `BatchNormalization`.
- `MultiGPU.strategy` фиксируется в конструкторе; менять число видимых GPU
  после создания объекта бессмысленно.
- `TFInit` не выбирает GPU и не ограничивает их видимость — используйте
  `CUDA_VISIBLE_DEVICES` до импорта TensorFlow.
