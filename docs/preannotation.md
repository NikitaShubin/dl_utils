# PreAnnotation: готовые сценарии предразметки

Ноутбуки для автоматической разметки изображений и видео с выгрузкой в CVAT.
Внутри — те же модули `dl_utils`, но собранные в интерактивный процесс.

Страницы модулей: [cvat](modules/cvat/README.md),
[cvat_srv](modules/cvat_srv/README.md), [ul_utils](modules/ul_utils.md),
[sam2al](modules/sam2al.md), [sam3al](modules/sam3al.md),
[ipy_utils](modules/ipy_utils.md).

## Запустить

В основном Docker-контейнере `dl_utils` откройте ноутбук из
`PreAnnotation/` и выполните первую ячейку: она добавляет `./..` и `./src` в
`sys.path`.

Отдельный контейнер для предразметки:

```bash
PreAnnotation/restart.sh
```

JupyterLab будет доступен на порту `573`, пароль — `PreAnnotator`. Папка
`PreAnnotation/project/` монтируется в `/workspace/project`, поэтому входные
файлы и результаты удобно держать в ней.

Оба Docker-образа уже содержат SAM 2, SAM 3, `decord` и веса моделей. Вне
Docker зависимости и путь к весам настраиваются вручную: для SAM 2 это
описано в [sam2al](modules/sam2al.md), для SAM 3 — в
[sam3al](modules/sam3al.md).

## Detection.ipynb

Детекция и трекинг. Ноутбук поддерживает два варианта:

- YOLO + Ultralytics — быстрый, работает с COCO-классами
  ([ul_utils](modules/ul_utils.md));
- Grounding DINO + BoxMOT — открытая детекция по текстовым запросам, медленнее
  ([gdinoal](modules/gdinoal.md), [boxmot_utils](modules/boxmot_utils.md)).

Типовой поток:

```python
from cvat import dir2unlabeled_tasks, tasks_auto_annottation
from ul_utils import UltralyticsModel

tasks = dir2unlabeled_tasks('./project/people.mp4', desc='Индексация')

img2df = UltralyticsModel('yolo26x.pt', tracker='bytetrack.yaml')
tasks = tasks_auto_annottation(
    tasks,
    img2df.img2df,
    label='person',
    desc='Предразметка',
)
```

Дальше ноутбук строит превью через `tasks2preview`, при необходимости
применяет `autofix_subtask` и `hide_skipped_objects_in_subtask`, затем
выгружает XML в проект CVAT через [`CVATSRV`](modules/cvat_srv/README.md).
Функции работы с задачами описаны в [cvat](modules/cvat/README.md).

## Segmentation.ipynb

Интерактивная сегментация видео через SAM 2. Модель и методы описаны в
[sam2al](modules/sam2al.md), виджет подсказок — в
[ipy_utils](modules/ipy_utils.md).

```python
from sam2al import SAM2

sam2 = SAM2(half=False)
sam2.load_video('project/people.mp4', offload_video_to_cpu=True)
```

Подсказки ставятся виджетом `IPYInteractiveSegmentation` или вручную:

```python
sam2.go_to(frame=120, obj_id='new', label='person')
sam2.points2masks(
    msk_points=[],
    box_points=[[100, 200, 300, 400]],
    pos_points=[[150, 250]],
    neg_points=[],
)
```

### SAM 2 → CVAT

1. Создайте проект CVAT и все нужные метки заранее, например
   `srv.new('Project 1', ['person', 'car'])`, где `srv` — подключение
   [`CVATSRV`](modules/cvat_srv/README.md).
2. Поставьте подсказки на опорных кадрах.
3. Соберите подзадачу:
   `df, file, true_frames = sam2.build_subtask(fit_segments=True)`.
4. Выгрузите результат в проект CVAT: `sam2.export2cvat(project)`.

`export2cvat` не создаёт проект и не добавляет метки, а разметку одноимённой
задачи заменяет целиком. `Segmentation.ipynb` при источнике «задача CVAT»
поступает иначе: скачивает старую разметку, объединяет её с новой через
`concat_dfs(..., smart=True)` и отправляет XML через `task.set_annotations`.
Поэтому в этом режиме разметка дополняется, а не заменяется. Подробности — в
разделе [«Выгрузить в CVAT»](modules/sam2al.md#выгрузить-в-cvat).

Подсказки хранятся в TSV (по умолчанию рядом с видео), поэтому работу можно
прервать и продолжить.

## SAM3.ipynb

Сегментация через SAM 3 с текстовыми запросами. Ноутбук использует
`SAM3VideoSemanticPredictor` из Ultralytics и обёртку
[`UltralyticsModel`](modules/ul_utils.md); особенности SAM 3 описаны в
[sam3al](modules/sam3al.md):

```python
from ultralytics.models.sam import SAM3VideoSemanticPredictor

from ul_utils import UltralyticsModel

predictor = SAM3VideoSemanticPredictor(overrides={
    'conf': 0.25,
    'task': 'segment',
    'mode': 'predict',
    'model': '/home/user/models/sam3.1_multiplex.pt',
})

sam3 = UltralyticsModel(predictor, text=['person', 'car'])
df, file, true_frames = sam3.video2subtask('project/people.mp4')
```

Дальше ноутбук переводит запросы в короткие метки через
[`LabelsConvertor`](modules/labels.md), превращает сегменты в прямоугольники,
отбрасывает короткие треки и выгружает результат в CVAT.

## Общие детали

- `dir2unlabeled_tasks` принимает файл, папку или список путей. Видео
  становится отдельной задачей, изображения группируются по папкам.
- [`CVATSRV`](modules/cvat_srv/README.md) спрашивает адрес сервера, логин и
  пароль при создании.
- `Detection.ipynb` создаёт проект, если его нет, и проверяет, что все нужные
  метки в нём присутствуют.
- `Segmentation.ipynb` для локального файла пишет в уже существующий проект;
  для задачи CVAT он обновляет разметку этой задачи.
- Долгие операции обёрнуты в `TimeIt` и `Beep`: по завершении раздаётся
  звуковой сигнал.

## Ограничения

- Ноутбуки рассчитаны на интерактивный запуск и не являются библиотечными
  модулями; переиспользуйте их код через соответствующие страницы модулей.
- Для SAM 2, SAM 3 и Grounding DINO нужен GPU и скачанные веса. Путь к SAM 2
  задаётся через `SAM2(model_path=...)`, к SAM 3 — через `overrides['model']`;
  значения в ноутбуках — примеры, их нужно заменить.
- `Segmentation.ipynb` хранит кадры видео во временной папке; для длинных
  записей нужно много диска.
- `Segmentation.ipynb` при записи локального файла требует существующий проект
  CVAT со всеми нужными метками; `export2cvat` не создаёт проект.
- `Detection.ipynb` не перезаписывает одноимённую задачу и выбрасывает
  `ValueError`; `sam2al.export2cvat` в такой ситуации заменяет разметку.
- Ноутбуки не покрыты автотестами; проверяйте результат на коротком фрагменте
  перед запуском на всей записи.
