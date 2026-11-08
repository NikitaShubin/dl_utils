# seg.py — сегменты и сегментационные датасеты

Фильтры списков масок, подгонка контуров и сохранение CVAT-задач в датасет
`inp`/`out` для обучения сегментации.

**Зависимости:** `numpy`, `opencv-python`, `pandas`, `tqdm`, `numba`;
файлы [`cv_utils.py`](cv_utils.md), [`cvat.py`](cvat/README.md),
[`utils.py`](utils/README.md). Через `cvat.py` нужен и
[`video_utils.py`](video_utils.md); внешние пакеты этих файлов ставятся
отдельно. Для примера сохранения нужен ещё [`labels.py`](labels.md) с
`openpyxl`; сам `seg.py` его не импортирует.

## Фильтровать маски

```python
from seg import DropByArea, Morphology

masks = Morphology('close', kernel=5)(masks)
masks = DropByArea(min_area=0.001, max_area=0.4)(masks)
```

`DropByArea` понимает целые числа как пиксели, а доли от 0 до 1 — как часть
площади изображения. `Morphology` сглаживает или утончает контуры.

`masks` — список объектов [`cv_utils.Mask`](cv_utils.md). Получить его из
CVAT-датафрейма можно через [`df2masks`](cvat/dataframes.md); фильтры не
принимают голые numpy-массивы.

## Убрать дубликаты

```python
from seg import DropDuplicates

masks = DropDuplicates(max_iou=0.85)(masks)
```

Фильтр сравнивает пары масок по площади и IoU и оставляет одну из почти
совпадающих.

## Разложить вложенные маски

```python
from seg import Cutter, Decompose

masks = Cutter(min_overlap=0.9)(masks)
masks = Decompose(min_overlap=0.9, steps=2)(masks)
```

`Cutter` вычитает меньшие сегменты из больших. `Decompose` перебирает
варианты вычитания и может сильно увеличить число масок.

## Выбрать главный сегмент

```python
from seg import MajorSegment

masks = MajorSegment(mask_of_interest)(masks)
```

Оставляется маска с максимальным IoU к шаблону; вложенные в неё меньшие
сегменты удаляются, а из больших она вычитается.

## Подогнать контуры в датафрейме

```python
from seg import fit_segments_in_df

df = fit_segments_in_df(
    df,
    imsize=(1080, 1920),
    drop_small_parts=True,
    fuse_by_groups=True,
)
```

Функция делает так, чтобы каждый пиксель принадлежал ровно одному сегменту.
`fuse_by_groups=True` объединяет сегменты одной группы и метки в один объект.
Работает с датафреймами без интерполяции или после
[`interpolate_df`](cvat/processing.md).

## Сохранить датасет

```python
from cvat import cvat_backups2tasks
from labels import LabelsConvertor
from seg import save_tasks2segmentation_dataset

# Задачи из распакованных бекапов CVAT:
tasks = cvat_backups2tasks('backups')

# Делим по задачам, а не по кадрам, чтобы кадры одного видео не попали
# одновременно в train и test:
train_tasks, val_tasks, test_tasks = tasks[:8], tasks[8:9], tasks[9:]

labels_convertor = LabelsConvertor(
    'labels.xlsx',
    'superlabels.xlsx',
    main_dict='label2superind',
)

subsets = [
    ('train', train_tasks),
    ('val', val_tasks),
    ('test', test_tasks),
]
for subset, subset_tasks in subsets:
    save_tasks2segmentation_dataset(
        subset_tasks,
        f'dataset/segmentation/{subset}',
        labels_convertor,
        make_preview=True,
        desc=f'Сохранение {subset}',
    )
```

`tasks` — список задач из распакованных бекапов, а не серверные объекты
[`CVATSRV`](cvat_srv/README.md). Как получить бекапы и разобрать их, описано в
[cvat: бекапы](cvat/backups.md). Разбиение на подвыборки делайте на уровне
задач или сцен: иначе кадры одного видео попадут в обучение и проверку.

`labels_convertor` должен иметь метод `apply2df(df)` и заменить строковые
`label` на числовые номера суперклассов. Готовая реализация —
[`LabelsConvertor`](labels.md) с `main_dict='label2superind'`. Функция рисует
маску по числу `superind + 1`. Значения `-1` (неиспользуемый класс) не
попадают в маску, а кадры с `-2` (запрещённый класс) пропускаются.

В каждой папке появятся `inp/` с изображениями, `out/` с масками классов и,
если `make_preview=True`, `prv/` с цветными превью. Имена файлов — сквозные
номера семплов.

В маску записывается `superind + 1`: `0` — фон, объектные классы начинаются с
`1`. Поэтому `num_classes` должно быть на единицу больше максимального значения
по всему `out/`, а не равно числу объектных классов и не максимуму одной маски:

```python
from pathlib import Path

import cv2

out_dir = Path('dataset/segmentation/train/out')
max_label = max(
    int(cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE).max())
    for mask_file in out_dir.glob('*.png')
)
print(max_label)  # num_classes = max_label + 1
```

Дальше датасет читает [`SegDataLoader`](keras_utils/data.md).

## Построить превью готового датасета

```python
from seg import make_seg_dataset_colored_preview

make_seg_dataset_colored_preview(
    'dataset/segmentation',
    num_classes=5,
    make_video=True,
)
```

Функция обходит подпапки `train/`, `val/` и `test/` и перерисовывает `prv/`.
`num_classes` — число значений маски, включая фон: для маски со значениями
`0..4` это `5`.

## Ограничения

- Фильтры `seg` рассчитаны на объекты `cv_utils.Mask`; обычные numpy-массивы
  нужно обернуть заранее.
- `DropByArea` с двумя `None` бросает `ValueError`: нужен хотя бы один порог.
- `fit_segments_in_df` требует размер изображения и не угадывает его из данных.
- `save_tasks2segmentation_dataset` ожидает задачи из распакованных бекапов в
  формате CVAT `(df, file, true_frames)`, а не серверные объекты `CVATSRV`.
  Конвертор должен реализовать `apply2df(df)`; фон в маске — `0`, объектные
  классы — `1..N`.
- `make_seg_dataset_colored_preview` ищет подпапки `train`, `val`, `test`; если
  какой-то из них нет, соответствующая часть пропускается.
- `Decompose` экспоненциально растёт по числу масок; ограничивайте `steps`.
