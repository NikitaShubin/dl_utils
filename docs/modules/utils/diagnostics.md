# utils: диагностика и граф зависимостей

Разбор импортов через AST, построение Mermaid-графа модулей и сравнение
вложенных структур.

## Сравнить две структуры

```python
from utils import obj_diff

expected = {'labels': ['car', 'person'], 'frames': 100}
actual = {'labels': ['car'], 'frames': 100}

print(obj_diff(expected, actual))
# Несовпадение размеров ['labels']: 2 != 1!
```

`obj_diff` рекурсивно обходит словари, списки, кортежи и множества и
возвращает читаемый путь до первого расхождения. Пустая строка означает
полное совпадение.

## Найти импорты файла

```python
from utils import ImportVisitor

visitor = ImportVisitor()
visitor.analyze_file('cvat_srv.py')

print(visitor.unconditional_imports)  # импорты верхнего уровня
print(visitor.conditional_imports)    # импорты внутри условий и функций
```

## Построить граф зависимостей репозитория

```python
from utils import draw_repo_dependency_graph

path = draw_repo_dependency_graph('.')
print(path)  # ./dependency_graph.mmd
```

Функция обходит все `.py`-файлы верхнего уровня, кроме начинающихся с `_`,
и записывает Mermaid-граф. Содержимое `dependency_graph.mmd` переносится в
`README.md`.

Рёбра — транзитивная редукция: `A → B` означает, что для `A` нужен `B`, а
окольные пути не дублируются. Пунктирное ребро соответствует условному
импорту.

## Разобрать граф вручную

```python
from utils import ImportVisitor

visitor = ImportVisitor()
graph = visitor.build_dependency_graph(['utils.py', 'cv_utils.py'])
reduced = visitor.filter_longest_paths(graph)

print(visitor.to_mermaid(reduced))
```

`build_dependency_graph` возвращает словарь с множествами `unconditional` и
`conditional` для каждого модуля. `filter_longest_paths` убирает короткие
пути, `to_mermaid` раскладывает узлы по уровням, а `to_dot` оставляет
Graphviz-вариант.

## Ограничения

- `ImportVisitor` смотрит только на синтаксис импортов: динамические
  `importlib.import_module('...')` в граф не попадают.
- Граф строится по файлам верхнего уровня; пакеты и вложенные папки не
  обходятся.
- `obj_diff` сравнивает значения через `==`; для NumPy-массивов это даёт
  массив булевых значений и может привести к неоднозначному результату.
