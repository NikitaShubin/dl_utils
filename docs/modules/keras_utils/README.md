# keras_utils.py — модели, слои и обучение Keras

Сборка сегментационных моделей из backbone-ов `keras.applications`, готовые
слои-блоки, метрики, колбеки и загрузчик датасета `inp`/`out`.

**Зависимости:** `keras` (или `tf_keras`, `tf.keras` — см. ниже), `tensorflow`,
`numpy`, `tensorboard`, `matplotlib`, `pandas`, `opencv-python`, `utils.py`.

## Выбрать реализацию Keras

Модуль читает переменную окружения `KERAS_MODULE` при импорте:

```python
import os

os.environ['KERAS_MODULE'] = 'tf_keras'  # keras | tf_keras | tf.keras | tfmot
from keras_utils import UNet
```

Значение по умолчанию — `keras`. Переменную нужно выставить до первого импорта
`keras_utils`; позже она не перечитывается. `tfmot_utils` делает это сам.

## Собрать модель сегментации

```python
from keras_utils import SegDataLoader, UNet

train_loader = SegDataLoader('dataset/train', num_classes=4, batch_size=8)

model = UNet(
    backbone='MobileNetV3Large',
    input_shape=(512, 512, 3),
    out_filters=4,
)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(train_loader, epochs=20)
```

`out_filters` — число классов, включая фон. В многоклассовом режиме оно должно
совпадать с `num_classes` загрузчика. При `out_filters > 2` выходная активация
`softmax`, иначе `sigmoid`; для бинарной сегментации берите `out_filters=1`,
`binary_crossentropy` и не задавайте `num_classes`. В бинарном режиме маска
хранится как `0/255`, а не `0/1`. Подробнее — [data.md](data.md) и
[models.md](models.md).

Если датасет создан через `seg.save_tasks2segmentation_dataset`, фон уже
занимает `0`, а объектные классы сдвинуты на `1`. Проверьте максимум по всему
`out/`, а не по одной маске, и возьмите `num_classes = max + 1`; в примере
выше маска содержит классы `0..3`.

Сохранение модели и получение маски из предсказания описаны в
[training.md](training.md).

## Разделы

- [Модели](models.md) — backbone-ы, `UNet`, `Deeplabv3Plus`, замена головы.
- [Слои](layers.md) — строительные блоки для собственных архитектур.
- [Обучение](training.md) — метрики, колбеки, история, дистилляция.
- [Данные](data.md) — `SegDataLoader` для папок `inp`/`out`.

## Как устроено

Модуль импортирует выбранный Keras и сразу выставляет короткие псевдонимы:
`layers`, `models`, `applications`, `callbacks`, `optimizers`, `losses`,
`backend`. Дальше код пользуется ими, не заботясь о том, какой именно пакет
подключён:

```python
from keras_utils import layers, models

inp = layers.Input((256, 256, 3))
out = layers.Conv2D(1, 1, activation='sigmoid')(inp)
model = models.Model(inp, out)
```

## Ограничения

- `KERAS_MODULE=tf.keras` и `KERAS_MODULE=tfmot` дают разные API; часть
  колбеков (`TrainingMode`, `AutoFinetune`) рассчитана на старый `learning_phase`
  и с Keras 3 не работает.
- Модели ожидают размер входа, кратный степени двойки числа пулингов
  backbone-а; иначе не совпадут размеры на skip-соединениях.
- `UNet` и `Deeplabv3Plus` не обучаются сами: нужен компилятор, лосс и
  датасет.
- Модуль импортирует `tensorboard` и `matplotlib` даже если обучение не
  запускается; в минимальном окружении их всё равно придётся поставить.
