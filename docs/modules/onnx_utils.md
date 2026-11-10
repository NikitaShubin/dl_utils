# 🔄 onnx_utils.py — ONNX-модели

Конвертация Keras в ONNX, упрощение до float16, динамическая и статическая
квантизация, а также инференс ONNX как обычной функции изображение → маска.

**Зависимости:** `onnx`, `onnxmltools`, `onnxruntime`, `onnxconverter-common`,
`numpy`, `tqdm`, `ml_utils.py`, `utils.py`; `tf2onnx` — только для
`keras2onnx`.

## 🔄 Конвертировать Keras-модель

```python
from onnx_utils import keras2onnx

f32, f16, dyn, stc = keras2onnx(
    model,
    f32='exports/model_f32.onnx',
    f16='exports/model_f16.onnx',
    dyn='exports/model_uint8.onnx',
    stc=None,
)
```

Функция сохраняет до четырёх вариантов и возвращает их объекты:

- `f32` — обычная float32-модель;
- `f16` — та же модель в float16;
- `dyn` — динамическая квантизация в uint8;
- `stc` — статическая int8-квантизация, требует калибровочный набор `ds`.

Чтобы отключить вариант, передайте `None` вместо пути.

## 🗜️ Статическая квантизация

```python
from onnx_utils import keras2onnx

f32, f16, dyn, stc = keras2onnx(
    model,
    f32='exports/model_f32.onnx',
    f16=None,
    dyn=None,
    stc='exports/model_int8.onnx',
    ds=calibration_dataset,  # коллекция пар (изображение, маска)
)
```

`DataReader` внутри создаёт `InferenceSession`, берёт имя входа и подаёт батчи
по одному. Калибровочных примеров должно быть достаточно, чтобы покрыть
диапазоны активаций.

## ⚡ Инференс ONNX-модели

```python
import cv2
from onnx_utils import ONNXModel

model = ONNXModel('exports/model_f16.onnx')
image = cv2.imread('frame_01.jpg')[..., ::-1]
mask = model(image)
```

`ONNXModel` сам определяет порядок осей входа и приводит изображение к нужному
типу. Результат возвращается в том же порядке осей, что и вход.

## 📥 Достать веса

```python
from onnx_utils import get_weights

weights = get_weights('exports/model_f32.onnx')
print([w.shape for w in weights])
```

## ⚠️ Ограничения

- `keras2onnx` требует `tf2onnx`. Без пакета модуль импортируется, но вызов
  бросает `ImportError`.
- `ONNXModel` поддерживает только float32 и float16 на входе; другие типы
  дают `ValueError`.
- Обёртка использует первый вход и первый выход модели. Многовходовые модели
  требуют собственного кода.
- Статическая квантизация пишет временный `tmp.onnx` и удаляет его только
  после успешного завершения.
