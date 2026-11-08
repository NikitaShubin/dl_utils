# cvat: датафрейм

Создание строк, изменение, сохранение и конвертация между датафреймом и
объектами `Mask`/`BBox`.

## Создать строку

```python
from cvat import CVATPoints, add_row2df, new_df

df = new_df()

points = CVATPoints.from_yolobbox(0.5, 0.4, 0.2, 0.3, imsize=(1080, 1920))
df = add_row2df(
    df,
    label='car',
    frame=0,
    true_frame=0,
    points=points.flatten(),
    type='rectangle',
)
```

`add_row2df` заполняет незаданные поля значениями по умолчанию и приводит типы
столбцов. Если `df` не передан, возвращается новый датафрейм с одной строкой.

## Изменить строку

```python
from cvat import update_row

row = df.iloc[0]
row = update_row(row, label='truck', outside=True)
df.iloc[0] = row
```

`update_row` не меняет исходную строку, если `inplace=False` (по умолчанию).
Незнакомые столбцам ключи попадают в `attribs`.

## Сохранить и загрузить

```python
from cvat import df_load, df_save

df_save(df, 'annotations.tsv')
same_df = df_load('annotations.tsv')
```

Формат — TSV. Списки и словари сериализуются в строки и восстанавливаются при
чтении; `track_id=None` переживает сохранение.

## Объединить датафреймы

```python
from cvat import concat_dfs

full_df = concat_dfs([df_1, df_2])

# Сдвинуть track_id, чтобы треки из разных датафреймов не пересекались:
full_df = concat_dfs([df_1, df_2], smart=True)
```

`concat_dfs` принимает датафреймы, строки и вложенные списки. `smart=True`
перенумеровывает `track_id` перед объединением.

## Конвертировать в объекты

```python
from cvat import df2masks, df2objs

masks = df2masks(df, imsize=(1080, 1920))
objects = df2objs(df, imsize=(1080, 1920))
```

`df2objs` выбирает `Mask` для полигонов и `BBox` для прямоугольников.
`df2masks` с `saving_memory=True` создаёт маски лениво.

## Конвертировать обратно

```python
from cvat import objs2df

df = objs2df(objects, label='car', frame=0)
```

Дополнительные аргументы становятся значениями столбцов; атрибуты объектов
добавляются из `obj.attribs`.

## Получить `annotations.json`

```python
from cvat import df2annotations
from utils import obj2json

annotations = df2annotations(df)
obj2json(annotations, 'annotations.json')
```

Словарь содержит теги, формы и треки в формате CVAT. Обратное чтение
выполняется парсером бекапа.

## Проверить корректность

```python
from cvat import check_df

check_df(df, file='data/video.mp4', true_frames={0: 0, 1: 1},
         labels=['car', 'person'])
```

Функция печатает записи с неизвестными метками, выходом за границы кадра и
другими противоречиями. Это отчёт, а не исключение.

## Ограничения

- `df2bboxes` в текущей версии содержит ошибку и возвращает не то, что
  ожидается; используйте `df2objs` или `df2masks`.
- `df_save` сохраняет только один датафрейм; для задачи используйте по файлу
  на подзадачу.
- `concat_dfs(smart=True)` меняет только `track_id`, но не `frame`; кадры
  разных подзадач могут пересекаться.
- `df2annotations` не создаёт `manifest.jsonl` и не копирует файлы — это
  делает код сборки бекапа.
