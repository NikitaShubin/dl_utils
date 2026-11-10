# 🎓 keras_utils: обучение

Метрики для сегментации, работа с историей, колбеки для прогрева, тонкой
настройки и аккуратной остановки, дистилляция.

## 🚀 Примеры

В примерах `train_ds`, `val_ds` и `test_ds` — любые Keras-совместимые
датасеты: `SegDataLoader` или результат `tf_utils.get_datasets`.

```python
# Добавить F-меру в метрики обучения
from keras_utils import MaxFscore

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[MaxFscore('F+TH', beta=1)],
)
```

```python
# Обучить с прогревом и тонкой настройкой
from keras_utils import AutoFinetune, Warmup

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[
        Warmup(steps=1000),
        AutoFinetune(monitor='val_loss', patience=5),
    ],
)
```

```python
# Сохранить историю и показать графики
from keras_utils import KerasHistory

history = model.fit(train_ds, validation_data=val_ds, epochs=20)
kh = KerasHistory(history)
kh.save('logs/hist.yaml')
kh.show()
```

```python
# Обучить студента по логитам учителя
from keras_utils import Distiller

distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer='adam',
    metrics=['accuracy'],
    student_loss_fn='sparse_categorical_crossentropy',
    distillation_loss_fn='kld',
    alpha=0.1,
    temperature=3,
)

distiller.fit(train_ds, epochs=20)
```

Модели описаны в разделе [«Модели»](models.md), загрузчик `inp`/`out` — в
[«Данные»](data.md).

## 📏 Метрики

### Измерить максимальную F-меру

```python
from keras_utils import MaxFscore

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[MaxFscore('F+TH', beta=1)],
)
```

`mode='F'` возвращает только максимальную F-меру, `'TH'` — оптимальный порог,
`'F+TH'` — оба значения в виде комплексного числа. Последний режим удобен для
анализа, но не для штатного `ModelCheckpoint`: Keras не умеет сравнивать
комплексные метрики.

### F-мера для любых размерностей

```python
from keras_utils import FBetaScore

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[FBetaScore(beta=1, average='macro')],
)
```

`FBetaScore` сначала выравнивает тензоры в вектор, поэтому работает с
`(N, H, W)` и `(N, H, W, C)`, в отличие от `keras.metrics.FBetaScore`.

`categorical_crossentropy` выбран потому, что [`SegDataLoader`](data.md) с
`num_classes` отдаёт one-hot маску. Если вы используете свой загрузчик с
целочисленными масками без one-hot, замените лосс на
`sparse_categorical_crossentropy`.

### Метрики на тестовой выборке

```python
from keras_utils import TestCallback

test_cb = TestCallback(test_ds)
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[test_cb])
```

В лог каждой эпохи добавляются `test_loss` и остальные метрики с префиксом
`test_`. Для KerasTuner используйте `TestCallback.prepare_model(model, test_ds)`
и `TestCallback()` без аргументов.

## 🔁 Обучение

### Переключить режим обучения

```python
from keras_utils import TrainingMode

with TrainingMode(training_mode=False):
    predictions = model(x)
```

Контекст подменяет `learning_phase`, и слои с `training=None` начинают
работать в инференс-режиме. В Keras 3 `learning_phase` отсутствует — используйте
`training=False` в вызове модели.

### Тонкая настройка в два этапа

```python
from keras_utils import AutoFinetune

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[
        AutoFinetune(
            monitor='val_loss',
            patience=5,
            lr_on_finetune=1e-4,
        ),
    ],
)
```

Сначала модель обучается с замороженными слоями. Когда `EarlyStopping`
срабатывает, `AutoFinetune` размораживает всю модель, перекомпилирует её и
продолжает обучение. Второй этап останавливается уже окончательно.

`lr_on_finetune='same'` сохраняет прежнюю скорость, `None` оставляет текущую.
`do_on_restart` — функция, которая вызывается перед вторым этапом.

### Остановка без потери истории

```python
from keras_utils import EndToEndEarlyStopping

model.fit(
    train_ds,
    epochs=100,
    callbacks=[EndToEndEarlyStopping(monitor='val_loss', patience=5)],
)
```

Отличие от обычного `EarlyStopping`: лучшие веса и метрики сохраняются между
перезапусками `fit`, поэтому повторный вызов не сбрасывает прогресс.

### Аккуратно завершить по сигналу

```python
from keras_utils import CarefulStopping

stopper = CarefulStopping()

with stopper.context():
    model.fit(
        train_ds,
        epochs=100,
        callbacks=[stopper.get_callback('on_epoch_end')],
    )
```

`CarefulStopping` перехватывает `SIGTERM`, дожидается указанного события и
завершает обучение без трейсбека. Для `Ctrl+C` добавьте `signal.SIGINT` через
`DoOnKill`.

Прямое использование колбека:

```python
import signal

from keras_utils import DoOnKill

callback = DoOnKill(
    stop_on='on_epoch_end',
    stop_signals=[signal.SIGTERM, signal.SIGINT],
)
```

### Перекомпилировать после разморозки

```python
from keras_utils import recompile_model

model.trainable = True
recompile_model(model, make_train_function=True)
```

Функция повторно применяет сохранённую конфигурацию компиляции. Это нужно
после изменения `trainable`, иначе Keras может продолжить использовать старый
граф обучения.

### Сохранить модель и получить маску

```python
from keras_utils import models

model.save('unet.keras')
model = models.load_model('unet.keras', compile=False)

batch = next(iter(val_ds))[0]
probabilities = model.predict(batch)
masks = probabilities.argmax(axis=-1)
```

Для модели с `out_filters=1` порог применяется к единственному каналу:

```python
masks = (model.predict(batch)[..., 0] > 0.5).astype('uint8')
```

`.keras` — родной формат Keras 3; для старого `tf_keras` можно использовать
`.h5`. При загрузке без `compile=False` Keras попытается восстановить лосс и
метрики, поэтому для инференса проще отключить компиляцию.

## 💾 История

### Сохранить и показать историю

```python
from keras_utils import KerasHistory

history = model.fit(train_ds, validation_data=val_ds, epochs=20)
kh = KerasHistory(history)
kh.save('logs/hist.yaml')
kh.show()
```

Загрузка обратно:

```python
kh = KerasHistory.load('logs/hist.yaml')

# Или из CSVLogger:
kh = KerasHistory.from_csv('logs/history.csv')
```

`group()` собирает метрики в словарь `имя → {train, val, test}`, а `show()`
рисует графики и подсвечивает фоном изменение `lr`.

## 🎛️ Скорость обучения

### Прогрев скорости обучения

```python
from keras_utils import Warmup

model.fit(
    train_ds,
    epochs=10,
    callbacks=[Warmup(steps=1000)],
)
```

Первые `steps` батчей скорость линейно растёт от нуля до исходного значения.
`restart()` запускает прогрев заново в середине обучения.

## 📈 Графики

### Рисовать графики по ходу обучения

```python
from keras_utils import PlotHistory

model.fit(
    train_ds,
    epochs=50,
    callbacks=[PlotHistory('logs/hist.yaml', show=True)],
)
```

Колбек сохраняет историю и перерисовывает графики в конце каждой эпохи. В
ноутбуке это очищает вывод ячейки.

### TensorBoard

```python
from keras_utils import TensorBoardContext

with TensorBoardContext('logs/fit') as tb_callback:
    model.fit(train_ds, epochs=20, callbacks=[tb_callback])
```

При входе в контекст запускается TensorBoard и возвращается колбек. При выходе
печатается команда для повторного запуска; сам сервис не останавливается.

## 🎓 Дистилляция знаний

```python
from keras_utils import Distiller

distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer='adam',
    metrics=['accuracy'],
    student_loss_fn='sparse_categorical_crossentropy',
    distillation_loss_fn='kld',
    alpha=0.1,
    temperature=3,
)

distiller.fit(train_ds, epochs=20)
```

`Distiller` — модель Keras, внутри вызывающая студента. Учитель работает в
режиме инференса. Выходы обеих моделей должны быть логитами: `softmax` для
дистилляционного лосса применяется внутри.

## ⚠️ Ограничения

- `MaxFscore` наследует `keras.metrics.AUC` и рассчитан на бинарную
  сегментацию; для многоклассовой нужен отдельный экземпляр на класс.
- `TrainingMode` и `AutoFinetune` используют старый `learning_phase` и
  перекомпиляцию; с Keras 3 их поведение не гарантировано.
- `KerasHistory.show()` использует `matplotlib` и не подходит для серверного
  обучения без дисплея.
- `Warmup` меняет `optimizer.learning_rate` как атрибут; в некоторых версиях
  Keras это не обновляет переменную оптимизатора.
- `DoOnKill` вешает обработчики сигналов только в главном потоке.
- `TensorBoardContext` не закрывает сервис при выходе из контекста.
- `Distiller` требует, чтобы студент и учитель принимали одинаковый вход и
  выдавали логиты одинаковой размерности.
