# 🏷️ cvat: метки

`CVATLabel` и `CVATLabels` описывают схему классов CVAT: имена, цвета, типы,
атрибуты и вложенные подметки.

## ➕ Создать набор меток

```python
from cvat import CVATLabels

labels = CVATLabels(['car', 'person', 'traffic_light'])
print(labels.names())  # ['car', 'person', 'traffic_light']
```

Из строк создаются метки типа `any` с автоматически подобранными цветами.

## 📖 Прочитать `labels.json`

```python
from cvat import CVATLabels

labels = CVATLabels('project/labels.json')
for label in labels:
    print(label.name, label.color, label.type)
```

Файл — JSON-список словарей с полями `name`, `color`, `type`, `attributes`.
Такой файл удобно хранить рядом с проектом.

## 🏷️ Одна метка

```python
from cvat import CVATLabel

label = CVATLabel({
    'name': 'car',
    'color': '#ff0000',
    'type': 'any',
    'attributes': [
        {'name': 'occluded', 'input_type': 'checkbox'},
    ],
})

print(label.rgb)       # (255, 0, 0)
label.rgb = (0, 255, 0)
print(label.to_dict())
```

`to_dict()` возвращает словарь в формате CVAT. Цвет можно задать кортежем,
списком или числом.

## 🎛️ Радиокнопки в Jupyter

```python
from ipy_utils import IPYRadioButtons
from cvat import CVATLabels

labels = CVATLabels('project/labels.json')
IPYRadioButtons(
    **labels.get_ipy_radio_buttons_kwargs('Класс объекта:'),
    setter=print,
).show()
```

## 📋 Отсортировать и скопировать

```python
sorted_labels = labels.sort()
copy_labels = labels.copy()
```

`sort` возвращает новый набор, отсортированный по именам; вложенные подметки
сортируются тоже. `copy` делает глубокую копию.

## 📤 Экспортировать

```python
dicts = labels.to_dicts()
```

Результат можно записать через `utils.obj2json`, чтобы получить
`labels.json`.

## ⚠️ Ограничения

- `CVATLabel` строго проверяет ключи: незнакомое поле приводит к `ValueError`.
- Тип метки (`any`, `rectangle`, `polygon`, `mask` и другие) не проверяется
  при разметке; несоответствие обнаружится позже, в CVAT.
- `CVATLabels` не сохраняет файл сам: используйте `to_dicts()` и
  `obj2json`.
- Подметки (`sublabels`) поддерживаются, но их атрибуты не валидируются.
