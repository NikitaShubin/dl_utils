"""yolo.py
********************************************
*   Модуль для конвертации списка задач в  *
*               YOLO-датасет.              *
*                                          *
*                                          *
* Основные функции:                        *
*   sources2statistic_and_train_val_...    *
*   ...test_tasks - расщепляет данные      *
*       на train/val/test и формирует      *
*       статистику.                        *
*                                          *
*   gen_yaml - формирует yaml-файл         *
*       описания YOLO-датасета.            *
*                                          *
*   tasks2yolo - сохраняет заданную        *
*       подвыборку в YOLO-подвыборку.      *
*                                          *
********************************************.
"""

import os
import shutil

import cv2
import numpy as np
import pandas as pd

from cvat import (
    CVATPoints,
    flat_tasks,
    sort_tasks,
    split_image_and_labels2tiles,
)
from ml_utils import train_val_test_split
from utils import (
    ImReadBuffer,
    cv2_img_exts,
    cv2_vid_exts,
    df2img,
    draw_contrast_text,
    mkdirs,
    mpmap,
    obj2yaml,
    rmpath,
)

# Полный список видеофайлов, поддерживаемых YOLO для обработки:
yolo_vid_exts = {
    '.asf',
    '.avi',
    '.gif',
    '.m4v',
    '.mkv',
    '.mov',
    '.mp4',
    '.mpeg',
    '.mpg',
    '.ts',
    '.wmv',
    '.webm',
}

# Полный список изображений, поддерживаемых YOLO для обработки:
yolo_img_exts = {
    '.bmp',
    '.dng',
    '.jpeg',
    '.jpg',
    '.mpo',
    '.png',
    '.tif',
    '.tiff',
    '.webp',
    '.pfm',
}


class YOLOLabels:
    """Формирует объект, содержащий описание разметки одного изображения в
    формате YOLO.
    """

    def __init__(
        self,
        df: 'Датафрейм, содержащий сегменты для текущего кадра',
        mode: 'Режим ("box" или "seg")' = 'box',
        imsize: 'Размер изображения' = None,
    ) -> None:
        self.mode = mode
        self.imsize = imsize

        # Объект, содежращий описание разметки в формате YOLO:
        self.yolo_labels = []

        # Расщепляем размер изображения на составляющие:
        height, width = imsize

        # Проходим по всем строкам, где контуры не скрыты:
        sub_df = df[~df['outside'].astype(bool)][
            [
                'label',  # noqa: E712
                'type',
                'points',
                'rotation',
            ]
        ].values
        for label, type_, points, rotation in sub_df:
            # Точек в контуре должно быть 4 для прямоугольников и эллипсов и не
            # меньше 6 для многоугольников:
            if type_ in ['rectangle', 'ellipse']:
                assert len(points) == 4
            elif type_ == 'polygon':
                assert len(points) >= 6

            # Получаем параметры фигуры вокруг объекта:
            points = CVATPoints(points, type_, rotation, imsize=imsize)

            # Приводим фигуру к нужному формату:
            attr = 'yoloseg' if mode == 'seg' else 'yolobbox'
            yolo_points = getattr(points, attr)(height, width)

            # Добавляем эти параметры в список:
            self.yolo_labels.append((label, yolo_points))

    # Заменяет метки с помощью функции:
    def apply_label_func(self, label_func) -> None:
        self.yolo_labels = [
            (label_func(label), yolo_points) for label, yolo_points in self.yolo_labels
        ]

    # Пишет разметку в файл датасета:
    def save(self, file):
        # Флаг успешности сохранения:
        succeeded = True

        with open(file, 'w') as f:
            for label, yolo_points in self.yolo_labels:
                # Если текущий объект вообще размечен, то пишем строчку в файл:
                if label >= 0:
                    points_str = ' '.join(map(str, yolo_points))
                    f.write(f'{label} {points_str}\n')

                # Если текущий объект ИСКЛЮЧЁН, то надо прерываем сохранение
                # и снимаем флаг успешности сохранения:
                elif label < -1:
                    succeeded = False
                    break

        # Если сохранение прервано, то надо удалять файл:
        if not succeeded:
            rmpath(file)

        # Возвращаем флаг успеха записи:
        return succeeded

    def draw_labels(self, image=None, edge_size=3, alpha=0.2):
        """Наносит метки Yolo-формата на изображение для превью."""
        # Размеры изображения:
        imsize = image.shape[:2]

        # Создаём изображение, если не задано:
        if image is None:
            image = np.zeros([*list(imsize), 3], np.uint8)

        # Определяем требуемый формат данных:
        type_ = 'polygon' if self.mode == 'seg' else 'rectangle'

        # Пробегаем по всем объектам в кадре:
        for label, points in self.yolo_labels:
            try:
                # Конвертируем YOLO-координаты в CVAT-координаты:

                # Оборачиваем точки в CVAT-класс:
                yolo_points = CVATPoints(points, type_, imsize=imsize)
                # Переводим из YOLO-формата в обычный для визуализации:
                points2draw = yolo_points.yolo2cvat()

                # Задаём цвет описанной фигуры:
                if label == -1:
                    color = (0, 0, 255)  # Синий   для неиспользуемых объектов
                elif label > -1:
                    color = (0, 255, 0)  # Зелёный для обычных        объектов
                else:
                    color = (255, 0, 0)  # Красный для исключённых    объектов
                # Если исключения работают корректно, то красный цвет не должен
                # вообще встречаться в превью!

                # Отрисовываем контуры, если надо:
                if edge_size:
                    image = points2draw.draw(image, color=color, thickness=edge_size)

                # Добавляем полупрозрачнаую заливку, если надо:
                if alpha:
                    image = points2draw.draw(
                        image, color=color, thickness=-1, alpha=alpha
                    )

                # Надписываем номер класса:
                image = points2draw.draw(image, str(label), thickness=0)

            except Exception:
                raise

        return image


def df2statistic(
    df: 'Анализируемый датафрейм',
    source_type: 'Тип источника данных',
    labels_convertor: 'Конвертор классов, меток и суперклассов',
    shapes_col_name: 'Имя столбца для объектов по кадрам' = 'shapes',
    tracks_col_name: 'Имя столбца для объектов по последовательностям' = 'tracks',
):
    """Подсчитывает статистику датафрейма подзадачи."""
    # Создаём датафреймы-счётчики:
    img_stat = labels_convertor.init_df_counter(source_type, shapes_col_name)
    # Статистика по каждому кадру в отдельности.
    vid_stat = labels_convertor.init_df_counter(source_type, tracks_col_name)
    # Статистика по видеопоследовательностям ("уникальные объекты").
    stat = pd.concat([img_stat, vid_stat], axis=1)
    # Объединённый датафрейм.

    unique_track_ids = set()  # Множество индексов уже учтённых объектов

    for raw in df.iloc:
        # Индекс объекта в пределах подзадачи:
        track_id = raw['track_id']

        # Метка объекта:
        label = raw['label']

        # Расшифровываем метку:
        meaning = labels_convertor.any_label2meaning(label)

        # Инкриментируем счётчик объектов в кадрах:
        stat.loc[meaning, shapes_col_name] += 1

        # Если объект следует добавить в статистику по
        # видеопоследовательностям:
        if track_id is None or track_id not in unique_track_ids:
            # Добавляем в статистику:
            stat.loc[meaning, tracks_col_name] += 1

            # Обновляем множество индексов уже учтённых объектов:
            unique_track_ids.add(track_id)

    return stat


def tasks2statistic(
    tasks: 'Анализируемый датафрейм',
    source_type: 'Тип источника данных',
    labels_convertor: 'Конвертор классов, меток и суперклассов',
    shapes_col_name: 'Имя столбца для объектов по кадрам' = 'shapes',
    tracks_col_name: 'Имя столбца для объектов по последовательностям' = 'tracks',
    desc: 'Статусбар' = None,
):
    """Считаем статистику для списка задач."""
    # Формируем список датафреймов:
    dfs = []
    for task in tasks:
        for df, _, _ in task:
            dfs.append(df)

    # Выполняем параллельную обработку:
    stats = mpmap(
        df2statistic,
        dfs,
        [source_type] * len(dfs),
        [labels_convertor] * len(dfs),
        [shapes_col_name] * len(dfs),
        [tracks_col_name] * len(dfs),
        desc=desc,
    )

    if len(stats) == 0:
        stats = [
            df2statistic(
                pd.DataFrame(),
                source_type,
                labels_convertor,
                shapes_col_name,
                tracks_col_name,
            )
        ]

    # Возвращаем объединённую статистику:
    return sum(stats)


def class_statistic2superclass_statistic(stat, labels_convertor):
    """Схлопывает статистику классов в статистику суперклассов."""
    # Целевая статистика-датафрейм:
    super_stat = [
        labels_convertor.init_df_counter('superclasses', column)
        for column in stat.columns
    ]
    # Список инициированных столбцов.
    super_stat = pd.concat(super_stat, axis=1)
    # Собираем отдельные столбцы в датафрейм.

    # Перебираем все строки исходного датафрейма:
    for row in stat.iloc:
        # Расшифровка класса:
        class_meaning = row.name

        # Расшифровка соответствующего суперкласса:
        superclass_meaning = labels_convertor.class_meaning2superclass_meaning.get(
            class_meaning.lower(), None
        )

        # Если текущему классу соответствует суперкласс, то добавляем
        # статистику:
        if superclass_meaning:
            super_stat.loc[superclass_meaning, :] += row

    return super_stat


def fill_skipped_rows_in_statistic(df, index):
    """Заполняет пропущенные объекты в статистике."""
    for meaning in index:
        if meaning not in df.index:
            df.loc[meaning, :] = [0] * len(df.columns)
    return df.astype(int)


def sources2statistic_and_train_val_test_tasks(
    source_name2tasks,
    yolo_ds_dir,
    labels_convertor,
    val_size=0.2,
    test_size=0,
    random_state=0,
):
    """Формирует статистику исходных данных и расщепляет на train/val/test
    # составляющие итогового датасета.
    """
    # Путь к файлам статистики:
    stat_dir = os.path.join(yolo_ds_dir, 'statistics')

    # Создаём папку статистики, если надо:
    mkdirs(stat_dir)

    # Шаблон строки описания статусбара:
    desc_template = 'Подсчёт статистики для %s выборки из %s-источника'

    """
    # Формируем полный список имён классов:
    meaning_list = sorted(list(set(
        labels_convertor.cvat_meanings_list + labels_convertor.gg_meanings_list
    )))
    """

    # Инициируем train/val/test составляющие датасета
    train_tasks, val_tasks, test_tasks = [], [], []

    # Инициируем общую статистику:
    full_class_train_stat = []
    full_class_val_stat = []
    full_class_test_stat = []
    full_superclass_train_stat = []
    full_superclass_val_stat = []
    full_superclass_test_stat = []

    # Перебираем все источники данных:
    for source_name, tasks in source_name2tasks.items():
        # Расщепляем текущие задачи на train/val/test:

        # Для CVAT-датасета разделение идёт по имени бекапа:
        if source_name == 'cvat':
            # Инициируем списки задач для выборок каждого типа:
            cur_train_tasks, cur_val_tasks, cur_test_tasks = [], [], []

            # Перебираем все задачи:
            for task in tasks:
                # Инициируем флаг принадлежности текущей задачи обучающей
                # выборке:
                is_train = None

                # Перебираем все подзадачи:
                for _, file, _ in task:
                    # Определяем имя бекапа для текущей подзадачи:
                    backup_dir_name = os.path.basename(
                        os.path.dirname(os.path.dirname(os.path.dirname(file)))
                    )

                    # Если вторым словом имени бекапа является "test":
                    if backup_dir_name[8:12].lower() == 'test':
                        # Если не для всех подзадач этой задачи имя бекапа
                        # содержит "test":
                        if is_train:
                            msg = (
                                'Именя бекапов отличаются в пределах одной '
                                f'задачи! Текущий файл: "{backup_dir_name}"'
                            )
                            raise ValueError(msg)

                        # Флагом помечаем текущую задачу как val:
                        is_train = False

                    # Если вторым словом имени бекапа НЕ является "test":
                    else:
                        # Если не для всех подзадач этой задачи имя бекапа
                        # НЕ содержит "test":
                        if is_train is False:
                            msg = (
                                'Именя бекапов отличаются в пределах одной '
                                f'задачи! Текущий файл: "{backup_dir_name}"'
                            )
                            raise ValueError(msg)

                        # Флагом помечаем текущую задачу как train:
                        is_train = True

                # В зависимости от значения флага вносим текующую задачу
                # в train- или val-список.
                (cur_train_tasks if is_train else cur_val_tasks).append(task)

        # Для CG-датасета всё уходит в обучающую выборку:
        elif source_name == 'cg':
            cur_train_tasks, cur_val_tasks, cur_test_tasks = tasks, [], []

        # Для остальных источников (включая GG) деление ведётся классическим
        # способом:
        else:
            cur_train_tasks, cur_val_tasks, cur_test_tasks = train_val_test_split(
                tasks, val_size=val_size, test_size=test_size, random_state=random_state
            )

        # Расфасовывающие текущие составляющие в общий train/val/test:
        train_tasks += cur_train_tasks
        val_tasks += cur_val_tasks
        test_tasks += cur_test_tasks

        # Подсчёт статистики классов:
        class_train_stat = tasks2statistic(
            cur_train_tasks,
            source_name,
            labels_convertor,
            desc=desc_template % (' обучающей ', source_name),
        )
        class_val_stat = tasks2statistic(
            cur_val_tasks,
            source_name,
            labels_convertor,
            desc=desc_template % ('проверочной', source_name),
        )
        class_test_stat = tasks2statistic(
            cur_test_tasks,
            source_name,
            labels_convertor,
            desc=desc_template % (' тестовой  ', source_name),
        )

        class_total_stat = class_train_stat + class_val_stat + class_test_stat

        full_class_train_stat.append(class_train_stat)
        full_class_val_stat.append(class_val_stat)
        full_class_test_stat.append(class_test_stat)

        # Схлопывание статистики в суперклассы:
        superclass_train_stat = class_statistic2superclass_statistic(
            class_train_stat, labels_convertor
        )
        superclass_val_stat = class_statistic2superclass_statistic(
            class_val_stat, labels_convertor
        )
        superclass_test_stat = class_statistic2superclass_statistic(
            class_test_stat, labels_convertor
        )
        superclass_total_stat = class_statistic2superclass_statistic(
            class_total_stat, labels_convertor
        )

        full_superclass_train_stat.append(superclass_train_stat)
        full_superclass_val_stat.append(superclass_val_stat)
        full_superclass_test_stat.append(superclass_test_stat)

        """
        total = []
        total.append(superclass_train_stat.rename({
            'shapes':f'train_{source_name}_shapes',
            'tracks':f'train_{source_name}_tracks'
        }, axis='columns'))
        total.append(superclass_val_stat  .rename({
            'shapes':  f'val_{source_name}_shapes',
            'tracks':  f'val_{source_name}_tracks'
        }, axis='columns'))
        total.append(superclass_test_stat .rename({
            'shapes': f'test_{source_name}_shapes',
            'tracks': f'test_{source_name}_tracks'
        }, axis='columns'))
        total.append(superclass_total_stat.rename({
            'shapes':f'total_{source_name}_shapes',
            'tracks':f'total_{source_name}_tracks'
        }, axis='columns'))
        general.append(pd.concat(total, axis=1))
        """

        # Cохраняем статистику классов и суперклассов в файлы:
        class_train_stat.to_csv(
            os.path.join(stat_dir, f'classes_train_{source_name}.cst'), sep='\t'
        )
        class_val_stat.to_csv(
            os.path.join(stat_dir, f'classes_val_{source_name}.cst'), sep='\t'
        )
        class_test_stat.to_csv(
            os.path.join(stat_dir, f'classes_test_{source_name}.cst'), sep='\t'
        )
        class_total_stat.to_csv(
            os.path.join(stat_dir, f'classes_total_{source_name}.cst'), sep='\t'
        )
        superclass_train_stat.to_csv(
            os.path.join(stat_dir, f'superclasses_train_{source_name}.cst'), sep='\t'
        )
        superclass_val_stat.to_csv(
            os.path.join(stat_dir, f'superclasses_val_{source_name}.cst'), sep='\t'
        )
        superclass_test_stat.to_csv(
            os.path.join(stat_dir, f'superclasses_test_{source_name}.cst'), sep='\t'
        )
        superclass_total_stat.to_csv(
            os.path.join(stat_dir, f'superclasses_total_{source_name}.cst'), sep='\t'
        )

        class_train_stat.to_excel(
            os.path.join(stat_dir, f'classes_train_{source_name}.xlsx')
        )
        class_val_stat.to_excel(
            os.path.join(stat_dir, f'classes_val_{source_name}.xlsx')
        )
        class_test_stat.to_excel(
            os.path.join(stat_dir, f'classes_test_{source_name}.xlsx')
        )
        class_total_stat.to_excel(
            os.path.join(stat_dir, f'classes_total_{source_name}.xlsx')
        )
        superclass_train_stat.to_excel(
            os.path.join(stat_dir, f'superclasses_train_{source_name}.xlsx')
        )
        superclass_val_stat.to_excel(
            os.path.join(stat_dir, f'superclasses_val_{source_name}.xlsx')
        )
        superclass_test_stat.to_excel(
            os.path.join(stat_dir, f'superclasses_test_{source_name}.xlsx')
        )
        superclass_total_stat.to_excel(
            os.path.join(stat_dir, f'superclasses_total_{source_name}.xlsx')
        )

        df2img(
            class_train_stat,
            os.path.join(stat_dir, f'class_train_stat_{source_name}.png'),
            'class_train_stat',
        )
        df2img(
            class_val_stat,
            os.path.join(stat_dir, f'class_val_stat_{source_name}.png'),
            'class_val_stat',
        )
        df2img(
            class_test_stat,
            os.path.join(stat_dir, f'class_test_stat_{source_name}.png'),
            'class_test_stat',
        )
        df2img(
            class_total_stat,
            os.path.join(stat_dir, f'class_total_stat_{source_name}.png'),
            'lass_total_stat',
        )
        df2img(
            superclass_train_stat,
            os.path.join(stat_dir, f'superclass_train_stat_{source_name}.png'),
            'superclass_train_stat',
        )
        df2img(
            superclass_val_stat,
            os.path.join(stat_dir, f'superclass_val_stat_{source_name}.png'),
            'superclass_val_stat',
        )
        df2img(
            superclass_test_stat,
            os.path.join(stat_dir, f'superclass_test_stat_{source_name}.png'),
            'superclass_test_stat',
        )
        df2img(
            superclass_total_stat,
            os.path.join(stat_dir, f'superclass_total_stat_{source_name}.png'),
            'superclass_total_stat',
        )

    return train_tasks, val_tasks, test_tasks


def gen_yaml(
    file_name: 'Имя сохраняемого файла',
    yolo_ds_dir: 'Путь до датасета (будет прописан в файл в неизменном виде!)',
    im_trn_dir: 'Изображения обучающей   выборки',
    im_val_dir: 'Изображения проверочной выборки',
    im_tst_dir: 'Изображения тестовой    выборки',
    superclasses: 'Словарь перехода от индекса суперкласса к его расшифровке',
):
    """Генерирует yaml-файл, описывающий датасет для YOLOv8."""
    # Сохранению подлежат только неотрицательные номера суперклассов:
    names = {k: v for k, v in superclasses.items() if k >= 0}

    # Формируем словарь, описывающий датасет:
    ds_yaml = {
        'path': os.path.abspath(yolo_ds_dir),  # Путь до датасета (абсолютный)
        'train': os.path.relpath(
            im_trn_dir, yolo_ds_dir
        ),  # Путь до изображений с обучающей   выборки относительно пути к датасету
        'val': os.path.relpath(
            im_val_dir, yolo_ds_dir
        ),  # Путь до изображений с проверочной выборки относительно пути к датасету
        'test': os.path.relpath(
            im_tst_dir, yolo_ds_dir
        ),  # Путь до изображений с тестовой    выборки относительно пути к датасету
        'names': names,
    }  # Словарь номер_класса -> расшифровка_класса

    # Если указан лищь путь до файла, имя добавляем сами:
    if os.path.isdir(file_name):
        file_name = os.path.join(file_name, 'data.yaml')

    # Сохраняем словарь в yaml-файл:
    return obj2yaml(ds_yaml, file_name)


def task2yolo(
    sample_ind: 'Число, с которого надо начать нумерацию семплов при сохранении в файлы',
    mode: 'Режим разметки. Один из {"box", "seg"}',
    task: 'Распарсеное фото/видео',
    labels_convertor: 'Конвертор классов, меток и суперклассов',
    images_dir: 'Папка для изображений',
    lablels_dir: 'Папка для разметок',
    preview_dir: 'Папка для изображений с рамками' = None,
    obj_scale: 'Масштабирование изображений по размеру объектов' = None,
    scale: 'Коэффициент(ы) масштабирования изображений' = 1,
    max_imsize: 'Максимальный размер изображения, после которого кадр разрезается' = (
        1080,
        1920,
    ),
):
    """Дописывает распарсенное фото/видео в датасет YOLO-формата.

    Тонкости параметра scale:
        Если scale задан одним числом, то этот масштаб применяется ко всему датасету.
        Если scale задан списком или кортежем - перебираются все содержащиеся в них масштабы.
        Если scale задан множеством, то из всех вариантов выбирется один для каждого кадра.
        Элементами списка scale могут выступать как числа, так и пары чисел. Пара воспринимается
        как интервал, из которого с равномерным распределением берётся случайное число.
    """
    # Если scale не список, и не множество, то делаем его списком:
    if isinstance(scale, tuple):
        scale = list(scale)
    if not isinstance(scale, (list, set)):
        scale = [scale]
    # Это нужно чтобы переменная стала итерируемой и копируемой.

    # Инициируем буфер чтения кадров:
    buffer = ImReadBuffer()

    # Перебор всех подзадач в рамках текущей задачи:
    for subtask_id, (df, file_path, true_frames) in enumerate(task):
        # Перебираем номера кадров прореженной последовательности, ...
        # ... пока не дошли до последнего размеченного в этой сцене кадра:
        for frame in sorted(true_frames.keys()):
            # Определяем номер текущего кадра исходной последовательности:
            true_frame = true_frames[frame]

            # Создаём датафрейм, содержащий только метки для текущего кадра:
            frame_df = df[df['true_frame'] == true_frame]

            # Если scale - множество, то выбираем каждый раз лишь одно
            # значение из этого множества:
            if isinstance(scale, set):
                cur_scale = [np.random.choice(list(scale))]

            # Если scale - список, то перебираем все масштабы:
            else:
                cur_scale = scale.copy()

            # Перебор всех коэффициентов масштабирования:
            for k in cur_scale:
                # Выбираем случайное значение k, если он задан списком:
                if hasattr(k, '__iter__'):
                    # Если список из 2 элементов, то это интервал:
                    if len(k) == 2:
                        # Берём случайное значение масштаба из заданного
                        # интервала:
                        k = np.random.rand() * abs(k[1] - k[0]) + min(k)

                    # Выводим ошибку, если элементов 3 и более:
                    elif len(k) > 2:
                        msg = (
                            'В подсписке масштабов не должно '
                            'быть больше 2 элементов, а имеем '
                            'k ='
                        )
                        raise ValueError(
                            msg,
                            k,
                        )

                # Читаем изображение через буфер:
                image = buffer(file_path, true_frame)

                # Определяем имя текущего входного файла:
                file = buffer.file
                # Брать приходится из buffer, т.к. он корректно работает в
                # случаее если вместо одного файла передан целый список.

                # Определяем расширение входного файла:
                inp_ext = os.path.splitext(file)[-1]

                # Формируем метки в YOLO-формате и применяем схлопывание
                # классов:

                # Делаем копию разметки для работы с текущим масштабом:
                scaled_frame_df = frame_df.copy()

                # Масштабируем изображение и разметку, если надо:
                if k != 1:
                    # Масштабируем изображение:
                    old_shape = np.array(image.shape[:2])
                    new_shape = (old_shape * k).astype(int)
                    image = cv2.resize(
                        image, new_shape[::-1], interpolation=cv2.INTER_AREA
                    )

                    # Масштабируем разметку:
                    chained_assignment = pd.options.mode.chained_assignment
                    pd.options.mode.chained_assignment = None
                    scaled_frame_df['points'] = scaled_frame_df['points'].apply(
                        lambda p: (CVATPoints(p) * k).flatten()
                    )
                    pd.options.mode.chained_assignment = chained_assignment
                    # При этом необходимо временно отключать предупреждения от
                    # pandas.

                    # Прописываем масштаб в соответствующий суффикс имён
                    # файлов:
                    scale_suffix = '_scale=' + str(k)

                # Если масштабирования нет, то соответствущий суффикс
                # оставляем незаполненным:
                else:
                    scale_suffix = ''

                # Разрезаем изображение на части, если задан маскимальный
                # размер:
                if max_imsize is None or mode == 'box':
                    tiles = [(scaled_frame_df, image)]
                else:
                    tiles = split_image_and_labels2tiles(
                        scaled_frame_df, image, max_imsize
                    )
                # Пока работает только в режиме обрамляющих прямоугольников!

                # Флаг копирования исходного изображения без конвертации и
                # пересжатия:
                is_image_copyable = (
                    k == 1 and inp_ext.lower() in cv2_img_exts and len(tiles) == 1
                )
                # Если входным файлом было изображение и оно не
                # масштабировалось и не разрезалось на части, то его можно
                # просто скопировать.

                # Задаём расширение выходного изображения:
                out_ext = inp_ext if is_image_copyable else '.jpg'

                # Сохраняем данные каждой части:
                for tile_ind, (tiled_df, tiled_img) in enumerate(tiles):
                    # Прописываем номер фрагмента изображения в имя файла,
                    # если фрагментов действительно несколько:
                    tile_suffix = ''
                    if len(tiles) > 1:
                        tile_suffix += '_tile№' + str(tile_ind)

                    # Пути к сохраняемым файлам:
                    general_name = f'%07d{scale_suffix}{tile_suffix}' % sample_ind
                    # Общая часть имён файлов (номер семлпла и cуффикс
                    # преобразований).
                    target_image_file = os.path.join(images_dir, general_name + out_ext)
                    # Имя файла изображения.
                    target_label_file = os.path.join(lablels_dir, general_name + '.txt')
                    # Имя файла разметки.
                    if preview_dir:
                        target_pview_file = os.path.join(
                            preview_dir, general_name + '.jpg'
                        )
                        # Имя файла предпросмотра.

                    # Если создаваемый файл изображения или разметки уже
                    # существует - выводим ошибку:
                    if os.path.isfile(target_image_file):
                        msg = (
                            f'Файл "{target_image_file}" '
                            'уже существует!\nОтносится к '
                            f'"{file}": {true_frame}.'
                        )
                        raise FileExistsError(msg)
                    if os.path.isfile(target_label_file):
                        msg = (
                            f'Файл "{target_label_file}" '
                            'уже существует!\nОтносится к '
                            f'"{file}": {true_frame}.'
                        )
                        raise FileExistsError(msg)

                    # Запись разметки:
                    yolo_labels = YOLOLabels(tiled_df, mode, tiled_img.shape[:2])
                    # Парсим датафрейм.
                    yolo_labels.apply_label_func(labels_convertor)
                    # Заменяем метки путём схлопывания в суперклас.
                    succeeded = yolo_labels.save(target_label_file)
                    # Записываем разметку в файл.

                    # Производим запись изображения, если разметка сохранилась
                    # успешно:
                    if succeeded:
                        # Просто копируем исходное изображение без пересжатия,
                        # если возможно:
                        if is_image_copyable:
                            shutil.copyfile(file, target_image_file)

                        # Сохраняем изображение с пересжатием, если
                        # копирование не возможно:
                        else:
                            cv2.imwrite(
                                target_image_file,
                                tiled_img,
                                [cv2.IMWRITE_JPEG_QUALITY, 95],
                            )

                    # Пропускаем дальнейшие действия, если семпл не был
                    # сохранён:
                    else:
                        continue
                    # Семпл не сохраняется если в кадр попал объект, который
                    # следует исключить.

                    # Запись превью, если надо:
                    if preview_dir is not None:
                        # Отрисовка всех меток:
                        tiled_img = yolo_labels.draw_labels(tiled_img)

                        # Наносим на превью кадра ещё и текст c доп.
                        # информацией:

                        # Если исходным файлом было видео:
                        if inp_ext.lower() in cv2_vid_exts:
                            # Извлекаем имя видео и датасета для надписи
                            img_path, img_name = os.path.split(file)
                            ds_name = os.path.basename(
                                os.path.abspath(os.path.join(img_path, '..', '..'))
                            )

                            # Формируем строку с номерами кадров:
                            frame_info = 'file=%d, frame=%s, true_frame=%d' % (
                                sample_ind,
                                frame,
                                true_frame,
                            )

                            # Формируем всю строку надписи:
                            caption = f'{ds_name}\n{img_name}\n{frame_info}'

                        # Если исходным файлом было изображение:
                        else:
                            # Извлекаем имя фото для надписи:
                            img_path, img_name = os.path.split(file)

                            # Извлекаем имена двух предыдущих папок:
                            dirs, dir2 = os.path.split(img_path)
                            dirs, dir1 = os.path.split(dirs)

                            dirs = (dir1,) if dir1 == dir2 else (dir1, dir2)

                            # Формируем всю строку надписи:
                            caption = '\n'.join((*dirs, img_name))

                        # Наносим надпись на изображение:
                        tiled_img = draw_contrast_text(tiled_img, caption)

                        # Сохраняем превью в файл:
                        cv2.imwrite(
                            target_pview_file, tiled_img, [cv2.IMWRITE_JPEG_QUALITY, 95]
                        )

            # Увеличиваем номер семпла:
            sample_ind += 1

    # Освобождаем ресурсы буфера:
    buffer.close()


def tasks2yolo(
    mode: 'Режим разметки. Один из {"box", "seg"}',
    tasks: 'Список распарсеных фото/видео',
    labels_convertor: 'Конвертор классов, меток и суперклассов',
    images_dir: 'Папка для изображений',
    lablels_dir: 'Папка для разметок',
    preview_dir: 'Папка для изображений с рамками' = None,
    obj_scale: 'Масштабирование изображений по размеру объектов' = None,
    scale: 'Масштабирование изображений' = 1,
    max_imsize: 'Максимальный размер изображения' = (1080, 1920),
    desc: 'Текст статус-бара' = 'Запись файлов',
):
    """Сохраняет распарсенные фото/видео в виде датасета YOLO-формата."""
    # Переводим каждую подзадачу в отдельную задачу для более эффективной
    # параллельной обработки:
    tasks = flat_tasks(tasks)

    # Сортируем задачи для унификации результата:
    tasks = sort_tasks(tasks)
    # Полезно для полной воспроизводимости проверочной выборки.

    # Создаём все недостающие папки:
    mkdirs(images_dir)
    mkdirs(lablels_dir)
    if preview_dir is not None:
        mkdirs(preview_dir)

    # Выходим сразу, если список задач пуст:
    if len(tasks) == 0:
        return []

    # Инициируем номер семпла для датасета:
    sample_ind = 0

    # Заполняем список аргументов для параллельной обработки:
    sample_inds = []
    modes = []
    labels_convertors = []
    images_dirs = []
    lablels_dirs = []
    preview_dirs = []
    obj_scales = []
    scales = []
    max_imsizes = []
    lens = []  # Количество изображений в каждой задаче
    for task in tasks:
        # Добавляем аргументы для текущей задачи в списки:
        sample_inds.append(sample_ind)
        modes.append(mode)
        labels_convertors.append(labels_convertor)
        images_dirs.append(images_dir)
        lablels_dirs.append(lablels_dir)
        preview_dirs.append(preview_dir)
        obj_scales.append(obj_scale)
        scales.append(scale)
        max_imsizes.append(max_imsize)

        # Сдвигаем стартовый номер семпла для следующей задачи на число кадров
        # в текущей задаче:
        cur_len = sum([len(true_frames) for _, _, true_frames in task])
        # Количество изображений в текущей задаче.
        lens.append(cur_len)
        sample_ind += cur_len

    # Сортируем задачи по убыванию количества изображений:
    if False:  # Режим отладки?
        task_ids = list(range(len(lens)))
        num_procs = 1
    else:
        task_ids = np.argsort(lens)[::-1]
        num_procs = 0
    sample_inds = [sample_inds[ind] for ind in task_ids]
    modes = [modes[ind] for ind in task_ids]
    tasks = [tasks[ind] for ind in task_ids]
    labels_convertors = [labels_convertors[ind] for ind in task_ids]
    images_dirs = [images_dirs[ind] for ind in task_ids]
    lablels_dirs = [lablels_dirs[ind] for ind in task_ids]
    preview_dirs = [preview_dirs[ind] for ind in task_ids]
    obj_scales = [obj_scales[ind] for ind in task_ids]
    scales = [scales[ind] for ind in task_ids]
    max_imsizes = [max_imsizes[ind] for ind in task_ids]

    # Параллельная сохранение данных:
    mpmap(
        task2yolo,
        sample_inds,
        modes,
        tasks,
        labels_convertors,
        images_dirs,
        lablels_dirs,
        preview_dirs,
        obj_scales,
        scales,
        max_imsizes,
        num_procs=num_procs,
        desc=desc,
    )

    return
