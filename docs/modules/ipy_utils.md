# 📓 ipy_utils.py — интерактив в Jupyter

Ссылки на файлы, виджеты и интерактивная сегментация прямо в ноутбуке.

**Зависимости:** `ipywidgets`, `numpy`, `matplotlib`, `mpl-point-clicker`,
`mpl-interactions`, `IPython`, `cv_utils.py`, `cvat.py`, `utils.py`.

## 🚀 Примеры

```python
# Дать ссылку на скачивание файла
from ipy_utils import path2link

path2link('results/preview.mp4', 'Скачать превью')
```

```python
# Показать видео в ноутбуке
from ipy_utils import show_video

show_video('results/preview.mp4', size=(480, 854))
```

```python
# Прочитать значение чекбокса
from ipy_utils import IPYCheckbox

fit_segments = IPYCheckbox(description='Подгонять контуры')
fit_segments.show()
print(fit_segments.value.val)
```

```python
# Запустить интерактивную сегментацию
from ipy_utils import IPYInteractiveSegmentation

viewer = IPYInteractiveSegmentation(**sam2.init_ipis_kwargs())
```

## 🔗 Ссылки и вывод

### Дать ссылку на файл

```python
from ipy_utils import path2link

path2link('results/preview.mp4', 'Скачать превью')
```

В JupyterLab появится ссылка на скачивание. Без второго аргумента текстом
будет имя файла.

### Объединить вывод ячеек

```python
from ipy_utils import concat

concat(output_1, output_2)
```

## 👁️ Медиа

### Показать видео

```python
from ipy_utils import show_video

show_video('results/preview.mp4', size=(480, 854))
```

`size` — высота и ширина в пикселях; можно передать одно число (высота) или
`None`, чтобы оставить исходный размер.

## 🎛️ Виджеты

### Чекбокс и кнопка

```python
from ipy_utils import IPYCheckbox, IPYButton

fit_segments = IPYCheckbox(description='Подгонять контуры')
fit_segments.show()

print(fit_segments.value.val)
```

`IPYCheckbox.value` — объект `Bool`, который обновляется при клике. В другой
ячейке читайте `fit_segments.value.val`.

```python
from ipy_utils import IPYButton

def run_preview():
    print('Превью строится...')

button = IPYButton('Построить превью', action=run_preview)
button.show()
```

По умолчанию кнопка очищает вывод ячейки перед вызовом действия.

### Радиокнопки

```python
from ipy_utils import IPYRadioButtons

mode = IPYRadioButtons(['preannotation', 'preview'], description='Режим')
mode.show()
```

С колбеком выбранное значение сразу передаётся в `setter`.

### Интерактивная сегментация

```python
from ipy_utils import IPYInteractiveSegmentation

sam2.go_to(frame=120, obj_id='new', label='person')
viewer = IPYInteractiveSegmentation(**sam2.init_ipis_kwargs())
```

Здесь `sam2` — загруженная модель из [sam2al](sam2al.md).
`init_ipis_kwargs()` передаёт текущий кадр `SAM2`, уже поставленные подсказки и
метод `points2masks`; конструктор сразу рисует виджет. В окне можно ставить
точки маски, рамки и положительные/отрицательные подсказки. `points2masks`
вызывается при каждом изменении и возвращает список масок, которые тут же
рисуются поверх изображения.

## ⚠️ Ограничения

- Модуль рассчитан на JupyterLab: `path2link` строит ссылку вида `/files/...`.
- `IPYInteractiveSegmentation` требует `ipympl` и интерактивный бэкенд
  `matplotlib`; в обычном скрипте виджет не появится.
- `show_video` читает файл с сервера, где запущен ноутбук; путь должен быть
  доступен этому процессу.
- `IPYCheckbox` хранит значение в обёртке `Bool`; не сравнивайте сам виджет с
  `True`.
