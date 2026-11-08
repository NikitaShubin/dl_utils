# utils: разное

Небольшие функции, которые не выросли в отдельные разделы: римские числа,
индексы и перестановки, случайные зёрна, круговой индекс.

## Римские числа

```python
from utils import rim2arabic

print(rim2arabic('XIV'))  # 14
print(rim2arabic('MCMXCIV'))  # 1994
```

Обратного преобразования нет: функция нужна для чтения римских обозначений в
именах файлов и папок.

## Индексы и перестановки

```python
from utils import invert_index_vector, reorder_lists

order = [1, 3, 0, 2]          # новая позиция каждого элемента
print(invert_index_vector(order))  # [2, 0, 3, 1]

names = ['car', 'person', 'bike']
scores = [0.9, 0.6, 0.8]
names, scores = reorder_lists(order, names, scores)
```

`invert_index_vector` строит обратную перестановку. Если индексы не
образуют полный диапазон `0..max`, в результате появятся `None` — это
ожидаемо.

## Списки произвольной вложенности

```python
from utils import flatten_list, unflatten_list

nested = [[1, 2], [3, [4, 5]]]
flat = flatten_list(nested)              # [1, 2, 3, 4, 5]
same = unflatten_list(flat, shape=[5])   # тот же плоский список
grid = unflatten_list(flat[:4], shape=[2, 2])
```

`flatten_list` принимает глубину: `depth=1` убирает только один уровень
вложенности. `unflatten_list` работает как `reshape` для списков; `-1`
вычисляется автоматически.

## Круговой индекс

```python
from utils import CircleInd

index = CircleInd(circle=4, ind=0)
print(index.inc())  # 1
print(index.dec())  # 0
print(index.dec())  # 3
```

`CircleInd` замыкается на границах и приводится к `int` — удобно для выбора
цвета или кадра по кругу.

## Воспроизводимость случайных чисел

```python
import random
from utils import InternalRandomSeed

seed = InternalRandomSeed(start_seed=42)

@seed
def random_color():
    return random.choice(['red', 'green', 'blue'])

print(random_color())  # всегда одинаково при одном start_seed
print(random.randint(0, 100))  # внешний генератор не сбит
```

`InternalRandomSeed` работает и как декоратор, и как контекстный менеджер.
Внутри блока используется собственное состояние `random`, внешнее остаётся
нетронутым.

## Привести значение к паре

```python
from utils import a2hw

print(a2hw(5))        # (5, 5)
print(a2hw([1, 2]))   # (1, 2)
```

`a2hw` дублирует одиночное значение и пропускает готовую пару. Полезен в
функциях, где ширина и высота могут быть заданы одним числом.

## Ограничения

- `rim2arabic` понимает только заглавные и строчные латинские обозначения
  `M, D, C, L, X, V, I`.
- `InternalRandomSeed` управляет только `random` из стандартной библиотеки, а
  не генераторами NumPy или PyTorch.
- `CircleInd` не поддерживает шаг больше единицы.
