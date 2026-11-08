# pt_utils.py — утилиты PyTorch

Выбор устройства, безопасная дисперсия, датасет сегментации и пара слоёв для
адресной передачи признаков.

**Зависимости:** `torch`, `numpy`, `opencv-python`.

## Выбрать устройство

```python
from pt_utils import AutoDevice

device = AutoDevice()                 # cuda, mps или cpu — что доступно
tensor = device.to(tensor)            # перенос на это устройство

print(device.device)                  # torch.device('cuda')
```

Можно задать устройство явно:

```python
device = AutoDevice('cpu')
```

`AutoDevice` при инициализации настраивает TF32 для GPU Ampere и новее.

## Загрузить датасет сегментации

```python
from pt_utils import SegDataset

dataset = SegDataset('dataset/train', num_classes=4)
image, mask = dataset[0]
print(image.shape, mask.shape)
```

Ожидается структура `dataset/train/inp` и `dataset/train/out` с одинаковыми
именами файлов. Если задан `num_classes`, маска возвращается в one-hot виде.
Перед кодированием маска делится на 255, поэтому в файлах должны быть только
значения `0` и `255`; индексы классов `0..N-1` не поддерживаются.

## Безопасная дисперсия

```python
from pt_utils import safe_var

variance = safe_var(tensor, dim=-1, default_value=0.0)
```

Обычный `torch.var` возвращает `NaN`, если элементов меньше двух. `safe_var`
подставляет `default_value` вместо такого результата.

## Слои Sender и Receiver

```python
import torch
from pt_utils import Receiver, Sender

sender = Sender(in_channels=64, out_channels=128, kernel_size=2)
receiver = Receiver(in_channels=128, out_channels=64, kernel_size=2)

message = sender(torch.randn(1, 64, 32, 32))
restored = receiver(message)
```

`Sender` повышает разрешение транспонированной свёрткой, `Receiver` понижает
его обычной свёрткой. Пара полезна в архитектурах, где признаки передаются
между ветками разного масштаба.

## Ограничения

- `AutoDevice.get_avliable_device` проверяет CUDA, затем MPS, затем CPU.
  Конкретный номер GPU не выбирается.
- `SegDataset` сопоставляет файлы по алфавиту и требует совпадения имён;
  лишние файлы с одной стороны не обнаруживаются.
- `safe_var` не заменяет `torch.var` в автотрассировке: ветвление по размеру
  тензора может быть невидимо для графа.
