# cvat.py — формат CVAT-задач

`cvat.py` — ядро работы с разметкой CVAT: единый датафрейм, геометрия контуров,
парсинг бекапов, постобработка и визуализация. Остальные модули репозитория
возвращают данные именно в этом формате.

**Зависимости:** `pandas`, `numpy`, `opencv-python`, `scipy`, `matplotlib`,
`Pillow`, `tqdm`, `cv_utils.py`, `video_utils.py`, `utils.py`.

## Модель данных

Задача CVAT — список подзадач. Каждая подзадача — кортеж из трёх элементов:

```python
subtask = (df, file, true_frames)
```

- `df` — `pandas.DataFrame` с разметкой;
- `file` — путь к видео или список изображений;
- `true_frames` — словарь `номер кадра в CVAT → номер кадра в исходном файле`.

Задача нужна, чтобы группировать подзадачи одного видео или сцены: при
разбиении на train/val/test единицей служит задача, а не отдельный кадр.

## Датафрейм

```python
from cvat import new_df

df = new_df()
print(df.columns.tolist())
```

Столбцы: `track_id`, `label`, `frame`, `true_frame`, `type`, `points`,
`occluded`, `outside`, `z_order`, `rotation`, `attributes`,
`mutable_attributes`, `attribs`, `group`, `source`, `elements`.

```python
import cv2
from cvat import CVATPoints, concat_dfs, new_df

frame = cv2.imread('frame_01.jpg')
points = CVATPoints.from_yolobbox(0.5, 0.4, 0.2, 0.3, frame.shape[:2])
row = points.to_dfrow()

df = concat_dfs([new_df(), row])
df['label'] = 'car'
```

## Собрать подзадачу вручную

```python
import cv2
from cvat import CVATPoints, add_row2df, new_df

image = cv2.imread('frame_01.jpg')
height, width = image.shape[:2]

df = new_df()
points = CVATPoints.from_yolobbox(0.5, 0.4, 0.2, 0.3, (height, width))
df = add_row2df(
    df,
    label='car',
    frame=0,
    true_frame=0,
    points=points.flatten(),
    type='rectangle',
)

subtask = (df, 'frame_01.jpg', {0: 0})
```

## Разделы

- [Датафрейм](dataframes.md) — создание строк, сохранение, конвертация в
  объекты и обратно.
- [Геометрия](geometry.md) — `CVATPoints`, `Triangle`, преобразования
  контуров.
- [Метки](labels.md) — `CVATLabel`, `CVATLabels`, цвета и атрибуты.
- [Бекапы и файлы](backups.md) — парсинг CVAT-бекапов, XML, неразмеченные
  задачи.
- [Постобработка](processing.md) — интерполяция, сцены, трекинг форм,
  склейка контуров.
- [Визуализация](visualization.md) — отрисовка кадров и превью-видео.
- [Авторазметка](autolabeling.md) — применение модели к задаче или папке
  бекапов.

## Как устроено

`CVATPoints` хранит координаты, тип фигуры и поворот. Методы `asrectangle`,
`aspolygon`, `crop`, `scale`, `rot90`, `flip` возвращают новые контуры, а
`to_dfrow` и `from_dfrow` связывают геометрию с датафреймом.

Парсер `cvat_backups2tasks` читает распакованный бекап: `task.json` и
`annotations.json` каждой задачи, собирает строки в датафрейм и выполняет
постобработку — интерполяцию, разбиение по сценам и удаление дублей.

## Ограничения

- Один датафрейм описывает одну подзадачу; смешивать несколько видео в одном
  `df` нельзя.
- `true_frames` должен содержать все используемые кадры, иначе интерполяция и
  превью пропустят часть данных.
- `df2annotations` и `subtask2xml` поддерживают не все типы CVAT; экзотические
  фигуры лучше проверять на тестовой задаче.
- Координаты хранятся в пикселях исходного изображения, а не в относительных
  единицах.
