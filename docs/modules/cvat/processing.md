# ⚙️ cvat: постобработка разметки

Интерполяция, разбиение по сценам, склейка многоконтурных объектов, трекинг
форм и вспомогательные операции над задачами.

## 🔄 Интерполировать контуры

```python
from cvat import interpolate_df, interpolate_tasks_df

df = interpolate_df(df, true_frames)

tasks = interpolate_tasks_df(tasks, desc='Интерполяция')
```

CVAT хранит только ключевые кадры. `interpolate_df` добавляет контуры для
промежуточных кадров, чтобы датафрейм был плотным. `interpolated_only=True`
возвращает только добавленные строки.

## ✂️ Разбить по сценам

```python
from cvat import split_true_frames_in_tasks

tasks = split_true_frames_in_tasks(tasks, desc='Разбиение по сценам')
```

Метки `scene-start` и `scene-finish` задают границы. Разбиение выполняется
после интерполяции; иначе границы могут попасть между ключевыми кадрами.

## 🔗 Склеить многоконтурные объекты

```python
from cvat import tasks_fuse_multipoly

tasks = tasks_fuse_multipoly(tasks, smart=True, desc='Склейка контуров')
```

CVAT хранит объект с дырками как один контур с повторяющимися точками.
`smart=True` аккуратнее сохраняет форму, `smart=False` просто убирает повторы.

## 🧹 Удалить дубли

```python
from cvat import drop_label_duplicates_in_tasks

tasks = drop_label_duplicates_in_tasks(tasks, desc='Удаление дублей')
```

Строки сравниваются по всем полям; полностью совпадающие записи удаляются.

## 🗑️ Удалить неразмеченные кадры

```python
from cvat import drop_unlabeled_frames

tasks = drop_unlabeled_frames(tasks)
```

Полезно, если в кадре не может не быть объектов, а разметчик пропустил
несколько кадров.

## ✏️ Исправить противоречия

```python
from cvat import autofix_df, hide_skipped_objects_in_df

df = autofix_df(df)
df = hide_skipped_objects_in_df(df, true_frames)
```

`autofix_df` убирает смену класса внутри одного трека и другие конфликты.
`hide_skipped_objects_in_df` ставит `outside=True` там, где объекта не было в
кадре, вместо интерполяции между далёкими ключами.

## 🎯 Трекинг несвязных форм

```python
from cvat import subtask_shapes2tracks

df, file, true_frames = subtask_shapes2tracks(
    subtask,
    minIoU=0.6,
    depth=20,
    untracked_label='unlabeled',
    desc='Трекинг форм',
)
```

Функция связывает покадровые детекции в треки по IoU. Полезна после
авторазметки, которая работает с отдельными кадрами. Ключевой кадр лучше
разметить вручную.

Для ключевого кадра в середине видео:

```python
from cvat import bidirectional_subtask_shapes2tracks

df, file, true_frames = bidirectional_subtask_shapes2tracks(subtask)
```

## 🧹 Применить фильтры к маскам

```python
from cvat import apply_mask_processing2df
from seg import DropByArea, Morphology

df = apply_mask_processing2df(
    df,
    imsize=(1080, 1920),
    processing=[Morphology('close', 5), DropByArea(min_area=100)],
    mpmap_kwargs={'desc': 'Фильтрация масок'},
)
```

Фильтры должны сохранять число и порядок масок: результат записывается обратно
в те же строки.

## ✂️ Разрезать большие кадры

```python
from cvat import split_image_and_labels2tiles

tiles = split_image_and_labels2tiles(
    df,
    image,
    max_im_size=(1080, 1920),
    area_part_th=0.5,
)
for tile_df, tile_image in tiles:
    ...
```

Изображение и разметка режутся на перекрывающиеся части. Метка сохраняется,
если после обрезки осталось не меньше `area_part_th` её площади.

## ✂️ Обрезать разметку

```python
from cvat import crop_df_labels

cropped = crop_df_labels(df, bbox=[100, 100, 900, 700], area_part_th=0.3)
```

## ⚠️ Ограничения

- `interpolate_df` требует полный `true_frames`; пропуски приведут к неверным
  индексам.
- Разбиение по сценам рассчитано на метки `scene-start`/`scene-finish`; другие
  имена игнорируются.
- `subtask_shapes2tracks` не различает классы: формы с разными метками могут
  попасть в один трек.
- `apply_mask_processing2df` проверяет только число масок, но не их порядок:
  если фильтр переставит маски, контуры попадут в чужие строки.
- `split_image_and_labels2tiles` не дедуплицирует объекты на пересечениях
  тайлов.
