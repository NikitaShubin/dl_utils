# 🏷️ labels.py — метки и суперметки

Конвертор между именами классов, их расшифровками, суперклассами и номерами
суперклассов. Читает Excel, CSV и TSV, умеет работать со словарями.

**Зависимости:** `pandas`, `treelib`, `openpyxl` (для `.xlsx`), `utils.py`.

## ✏️ Заменить метки по словарю

```python
from labels import LabelsConvertor

converter = LabelsConvertor({
    'person': 'people',
    'crowd': 'people',
    'car': 'transport',
    'plane': 'transport',
})

print(converter('person'))  # people
```

Экземпляр — обычный словарь: `converter['car']` тоже работает. Если метки нет
в словаре, будет `KeyError`.

## ⚙️ Применить к датафрейму

```python
import pandas as pd
from labels import LabelsConvertor

converter = LabelsConvertor({'car': 'transport', 'bus': 'transport'})
df = pd.DataFrame({'label': ['car', 'bus'], 'frame': [0, 0]})

converted = converter.apply2df(df)
print(converted['label'].tolist())  # ['transport', 'transport']
```

Исходный датафрейм не меняется: возвращается копия с заменённым столбцом
`label`.

## ⚙️ Применить к объектам

```python
from cv_utils import BBox
from labels import LabelsConvertor

converter = LabelsConvertor({'car': 'transport'})
objects = [BBox.from_yolo([0.5, 0.5, 0.2, 0.2], attribs={'label': 'car'})]

objects = converter.apply2objs(objects)
print(objects[0].attribs['label'])  # transport
```

Работает с любыми объектами, у которых есть `attribs['label']` и метод
`copy()`.

## 📥 Загрузить таблицу

```python
from labels import LabelsConvertor

# Таблица меток и таблица суперметок:
converter = LabelsConvertor('labels.xlsx', 'superlabels.xlsx')

# Только таблица суперметок:
converter = LabelsConvertor('superlabels.xlsx')
```

Поддерживаются `.xlsx`, `.csv`, `.tsv`, `.txt`. Ожидаемые столбцы:
`Метка в CVAT`, `Метка в другом источнике данных`,
`Наименование суперкласса`. Шаблоны лежат в корне репозитория:
`labels_template.xlsx` и `superlabels_template.xlsx`.

Для `.xlsx` нужен `openpyxl`. Старый `.xls` пересохраните в `.xlsx`: код
пытается открыть его через `openpyxl`, а тот этот формат не читает.

## 🎯 Выбрать итоговое представление

```python
converter = LabelsConvertor(
    'labels.xlsx',
    'superlabels.xlsx',
    main_dict='label2superind',
)
print(converter('car'))  # номер суперкласса, например 2
```

Доступные переходы:

- `label2superlabel` — имя метки в имя суперметки;
- `label2superind` — имя метки в номер суперметки (для YOLO и масок `seg`);
- `auto` — самый длинный переход, который позволяют данные.

## 🏷️ Удалить и запретить метки

```python
converter = LabelsConvertor(
    'labels.xlsx',
    'superlabels.xlsx',
    values2del=['ignore'],
    values2raise=['person'],
)
```

`apply2df` и `apply2objs` выбросят объекты с `values2del` и бросят
`ForbiddenLabelError` при `values2raise`. Прямой вызов `converter('person')`
запрет игнорирует.

## ✅ Проверить неизвестные метки

```python
unknown = converter.get_unknown_labels(df)
if unknown:
    print('Нет в словаре:', unknown)
```

## ⚠️ Ограничения

- `apply2df` меняет только столбец `label`; атрибуты внутри `attributes` не
  конвертируются.
- Цепочка `label → meaning → superlabel` должна быть согласованной: разные
  расшифровки одной метки дают ошибку.
- Для `.xlsx` нужен `openpyxl`; старый `.xls` не поддерживается, пересохраните
  его в `.xlsx`.
- `values2del` и `values2raise` не действуют при вызове экземпляра как функции.
