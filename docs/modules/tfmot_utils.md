# tfmot_utils.py — квантизация и прунинг

Quantization-aware training и прунинг через TensorFlow Model Optimization.
Модуль принудительно переключает `keras_utils` на `tf_keras`.

**Зависимости:** `tensorflow-model-optimization`, `keras_utils.py`,
`tensorflow`.

## Подготовить модель к квантизации

```python
from tfmot_utils import Quantize

quantized_model, prepared_model = Quantize()(model)

quantized_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
quantized_model.fit(train_ds, epochs=5)
```

`Quantize()` — функтор: он клонирует модель, помечает слои, которые нельзя
квантизировать, и собирает QAT-модель. Возвращаются два объекта:

- `quantized_model` — модель с псевдоквантизацией для обучения;
- `prepared_model` — промежуточная модель с аннотациями слоёв.

Параметры компиляции исходной модели переносятся автоматически.

## Загрузить сохранённую QAT-модель

```python
from tfmot_utils import Quantize

model = Quantize.load_model('models/qat_model.keras')
```

При загрузке подключаются кастомные конфигурации квантизации, иначе Keras не
найдёт классы.

## Прунинг

```python
from tfmot_utils import prune_low_magnitude, get_pruning_callbacks

pruned_model = prune_low_magnitude(model)
pruned_model.compile(optimizer='adam', loss='mse')

callbacks = get_pruning_callbacks('logs/pruning')
pruned_model.fit(train_ds, epochs=10, callbacks=callbacks)
```

`prune_low_magnitude` оборачивает только поддерживаемые слои, остальные
оставляет как есть. Колбеки обновляют маску прунинга и пишут события в
TensorBoard.

## Проверить приложения Keras

```python
from tfmot_utils import test_aplications

test_aplications(include_top=False)
```

Функция перебирает конструкторы `keras.applications`, пытается квантизировать
каждую модель и печатает причины ошибок. Полезно перед выбором backbone.

## Ограничения

- Модуль выставляет `KERAS_MODULE=tf_keras` при импорте. Если нужен обычный
  `keras`, импортируйте `keras_utils` раньше или не используйте этот модуль.
- `Quantize.apply_quantization` знает лишь несколько «безопасных» слоёв;
  новые слои могут потребовать собственной конфигурации.
- Прунинг и квантизация меняют структуру модели: после них нельзя просто
  загрузить исходные веса без обёрток.
- `test_aplications` строит все модели без весов `imagenet`, поэтому
  занимает заметное время и память.
