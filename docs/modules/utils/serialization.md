# utils: сериализация

JSON и YAML в одну строку, плюс `ManyToOne` — словарь с обратным индексом для
классификаторов и справочников.

## Прочитать и записать JSON/YAML

```python
from utils import obj2json, json2obj, obj2yaml, yaml2obj

config = {'project': 'Project 1', 'labels': ['car', 'person']}

obj2json(config, 'config.json')
obj2yaml(config, 'config.yaml')

same_json = json2obj('config.json')
same_yaml = yaml2obj('config.yaml')
```

`obj2json(..., compact=True)` пишет JSON без отступов — удобно для больших
файлов, которые читает только код.

## Классификатор «многие к одному»

`ManyToOne` хранит соответствие `ключ → значение` и умеет отдавать обратный
индекс `значение → [ключи]`:

```python
from utils import ManyToOne

classes = ManyToOne({
    'car': 'vehicle',
    'bus': 'vehicle',
    'cat': 'animal',
})

print(classes['car'])            # vehicle
print(classes.one2many['vehicle'])  # ['car', 'bus']
```

Инициализация из обратного словаря требует явного типа:

```python
from utils import ManyToOne

classes = ManyToOne(
    {'vehicle': ['car', 'bus'], 'animal': ['cat']},
    type='one2many',
)
print(classes['bus'])  # vehicle
```

## Сохранить классификатор

```python
from utils import ManyToOne

classes = ManyToOne({'car': 'vehicle', 'bus': 'vehicle', 'cat': 'animal'})

classes.save('classes.yaml')                    # компактно: vehicle: [car, bus]
classes.save('classes.json', type='many2one')   # развёрнуто: car: vehicle
```

Обратное чтение — обычный конструктор: тип определяется автоматически по
структуре файла.

## Нарисовать связи

```python
from utils import ManyToOne

classes = ManyToOne({'car': 'vehicle', 'bus': 'vehicle', 'cat': 'animal'})
classes.to_mermaid('classes.mmd')
```

Получится `graph LR` с рёбрами от ключей к значениям. Полезно вставить в
Markdown-документацию или README.

## Ограничения

- `obj2yaml` использует `yaml.safe_dump`: произвольные Python-объекты не
  сериализуются, только базовые типы.
- `ManyToOne` требует, чтобы каждый ключ принадлежал ровно одной группе;
  пересечения вызывают `KeyError`.
- `json2obj` и `yaml2obj` возвращают то, что записано в файле: корневым
  элементом не обязан быть словарь.
