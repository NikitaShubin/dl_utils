# 📈 ml_utils.py — общие ML-помощники

Пути к моделям, one-hot, перестановка осей тензора и разбиение выборки.

**Зависимости:** `numpy`, `scikit-learn`, `utils.py`.

## 🚀 Примеры

```python
# Найти модель в ~/models
from ml_utils import model2home

model_path = model2home('sam_hq_vit_h.pth')
print(model_path)  # /home/user/models/sam_hq_vit_h.pth
```

```python
# Сделать one-hot из масок
import numpy as np
from ml_utils import ohe

masks = np.array([[0, 1], [2, 0]], dtype=np.int64)
one_hot = ohe(masks, num_classes=3)
print(one_hot.shape)  # (2, 2, 3)
```

```python
# Переставить оси тензора и определить, где каналы
import numpy as np
from ml_utils import is_channel_first, nchw2nhwc

batch_nhwc = nchw2nhwc(np.zeros((1, 3, 224, 224)))
print(batch_nhwc.shape)  # (1, 224, 224, 3)
print(is_channel_first((3, 512, 1024)))   # True
```

```python
# Разбить выборку на train/val/test
from ml_utils import train_val_test_split

samples = list(range(100))
train, val, test = train_val_test_split(samples, val_size=0.2, test_size=0.1)
```

## 🔍 Поиск модели

### Найти модель в `~/models`

```python
from ml_utils import model2home

model_path = model2home('sam_hq_vit_h.pth')
print(model_path)  # /home/user/models/sam_hq_vit_h.pth
```

Если передан только имя файла, а такого файла нет в текущей папке, путь
достраивается до `~/models/<имя>`. Папка создаётся автоматически. Если файл
существует или путь содержит директории, он возвращается без изменений.

## 🔢 Работа с тензорами

### One-hot для масок

```python
import numpy as np
from ml_utils import ohe

masks = np.array([[0, 1], [2, 0]], dtype=np.int64)
one_hot = ohe(masks, num_classes=3)
print(one_hot.shape)  # (2, 2, 3)
```

### Переставить оси тензора

```python
from ml_utils import nchw2nhwc, nhwc2nchw

batch_nhwc = nchw2nhwc(batch_nchw)
batch_nchw = nhwc2nchw(batch_nhwc)
```

Доступны `nchw2nhwc`, `nhwc2nchw`, `chw2hwc`, `hwc2chw`. Все функции
возвращают представление без копирования, если позволяет раскладка памяти.

### Понять, где каналы

```python
from ml_utils import is_channel_first

print(is_channel_first((3, 512, 1024)))   # True
print(is_channel_first((512, 1024, 3)))   # False
```

Функция смотрит на форму: канал считается первым, если его размер меньше
остальных измерений.

## ✂️ Выборка

### Разбить выборку

```python
from ml_utils import train_val_test_split

train, val, test = train_val_test_split(samples, val_size=0.2, test_size=0.1)
```

`soft_train_test_split` делает то же для двух выборок. Обе функции — обёртки
`sklearn.model_selection.train_test_split` с дополнительными проверками
размеров.

## ⚠️ Ограничения

- `model2home` не скачивает модели: он только подставляет путь. Загрузку
  выполняет вызывающий код.
- `is_channel_first` — эвристика по форме, а не по смыслу тензора. Для
  маленьких изображений с большим числом каналов она может ошибиться.
- `ohe` возвращает `float32`; для экономии памяти приводите результат к
  `uint8` самостоятельно.
