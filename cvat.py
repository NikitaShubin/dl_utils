'''
********************************************
*         Работа с CVAT-датасетами.        *
*                                          *
* Из всех используемых сейчас датасетов    *
* (CVAT, GG и CG) формат, используемый в   *
* CVAT, является наиболее общим, в связи с *
* чем описанные здесь функции и классы     *
* активно используются в других парсерах.  *
*                                          *
* Сам CVAT-формат задач, который создают   *
* сейчас все используемые в проекте        *
* парсеры датасетов представляет собой     *
* список задач. Каждая задача так же       *
* представляет собой список уже подзачач,  *
* каждая из которых в свою очередь         *
* является списком или кортежем из трёх    *
* элементов:                               *
*                                          *
*   df - датафремй данных разметки;        *
*   file - путь к изображению/видео;       *
*   true_frames - словарь перехода от      *
*     номеров кадров в CVAT к номерам      *
*     кадров в видео (содержит только те   *
*     кадры, которые используются в данной *
*     подзадаче).                          *
*                                          *
* Задача разбивается на подзадачи,         *
* например, когда подзадачи являются       *
* фрагментами одного видеофайла, либо      *
* серия фотографий одного проекта/полёта.  *
* Т.е. подзадачи каким-либо образом имеют  *
* высокую общность фоноцелевых обстановок, *
* и не должны быть разбиты на обучающую и  *
* проверочную выборки. Для этого их и      *
* группируют в список подзадач, а на       *
* подвыборки разбиваются уже только сами   *
* задачи.                                  *
*                                          *
*                                          *
* Основные функции и классы:               *
*   type(self) - класс, инкапсулирующий    *
*       разметку одного объекта. Позволяет *
*       выполнять над разметкой ряд        *
*       преобразований, включя конвертацию *
*       в YOLO-формат и обратно, поворот,  *
*       масштабирование, интерполяцию и    *
*       т.п.                               *
*                                          *
*   cvat_backups2tasks - сам парсер,       *
*       читающий распакованные             *
*       CVAT-бекапы, превращая их в список *
*       задач CVAT-формата.                *
*                                          *
*   drop_bbox_labled_cvat_tasks - ф-ия,    *
*       выбрасывающиая из списка задач те  *
*       из них, что размечены описывающими *
*       прямоугольниками, а не сегментами. *
*                                          *
*   sort_tasks - ф-ия сортировки списка    *
*       задач по имени используемого файла *
*       (нужна для достижения              *
*       воспроизводимости результатов      *
*       конвертации)                       *
*                                          *
********************************************
'''


import os
import cv2
import json
import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import ImageColor
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from utils import (mpmap, ImReadBuffer, reorder_lists, mkdirs, CircleInd,
                   apply_on_cartesian_product, extend_list_in_dict_value,
                   DelayedInit, color_float_hsv_to_uint8_rgb,
                   draw_contrast_text, put_text_carefully, cv2_vid_exts,
                   cv2_img_exts, split_dir_name_ext, get_file_list, cv2_exts)
from cv_utils import Mask, build_masks_IoU_matrix
from video_utils import VideoGenerator, ViSave, recomp2mp4


# Словарь имён и типов полей, которые надо считать из разметки:
df_columns_type = {'track_id'  : object, # 
                   'label'     : str   , # 
                   'frame'     : int   , # 
                   'true_frame': int   , # 
                   'type'      : str   , # 
                   'points'    : object, # 
                   'occluded'  : bool  , # 
                   'outside'   : bool  , # 
                   'z_order'   : int   , # 
                   'rotation'  : float , # 
                   'attributes': object, # 
                   'group'     : int   , # 
                   'source'    : str   , # 
                   'elements'  : object} # 

# Словарь имён полей из разметки и их значений по умолчанию:
df_default_vals = {'track_id'  : None     , # 
                   'label'     : ''       , # 
                   'frame'     : 0        , # 
                   'true_frame': 0        , # 
                   'type'      : 'polygon', # 
                   'points'    : []       , # 
                   'occluded'  : False    , # 
                   'outside'   : False    , # 
                   'z_order'   : 0        , # 
                   'rotation'  : 0.       , # 
                   'attributes': []       , # 
                   'group'     : 0        , # 
                   'source'    : 'manual' , # 
                   'elements'  : []       } # 


def get_column_ind(df, column):
    '''
    Находит номер столбца в датафрейме по его имени.
    Полезно для индексации через df.iloc.
    '''
    return np.where(df.columns == column)[0][0]


def new_df():
    '''
    Создаёт новый датафрейм для видео/изображения:
    '''
    return pd.DataFrame(columns=df_columns_type.keys()).astype(df_columns_type)
    # Задавать тип столбцов нужно, чтобы при конкатенации строк не вылезали
    # предупреждения.


def add_row2df(df=None, **kwargs):
    '''
    Добавляет новую строку с заданнымии параметрами в датафрейм.
    Значения незаданных параметров берутся из df_default_vals.
    '''
    # Создаём датафремй, если он не задан:
    if df is None:
        df = new_df()

    # Создаём новую строку с параметрами по умолчанию:
    row = pd.Series(df_default_vals)

    # Заменяем дефолтные значения на входные параметы:
    for key, val in kwargs.items():
        row[key] = val

    # Превращаем строку в датафрейм с заданными типами столбцов
    row = pd.DataFrame(row).T.astype(df_columns_type)

    # Возвращаем объединённый (или только новый, если исходный не задан)
    # датафрейм:
    return row if df is None else pd.concat([df, row])


class DisableSettingWithCopyWarning:
    '''
    Контекст отключения предубпреждений SettingWithCopyWarning от pandas.

    Пример использования:
        mask = df['points'].notna()
        with DisableSettingWithCopyWarning():
            df.loc[mask, 'points'] = df.loc[mask, 'points'].apply(
                smart_fuse_multipoly_in_df)
    '''

    def __enter__(self):
        self.chained_assignment = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None
        return self

    def __exit__(self, type, value, traceback):
        pd.options.mode.chained_assignment = self.chained_assignment


def shape2df(shape    : 'Объект, из которого считываются данные в  первую очередь'       ,
             parent   : 'Объект, из которого считываются данные во вторую очередь' = {}  ,
             track_id : 'ID объекта'                                               = None,
             df       : 'Датафрейм, в который данные нужно добавить'               = None):
    '''
    Вносит в датафрейм инфу о новом объекте.
    '''
    # Список извлекаемых значений:
    columns = set(df_columns_type.keys()) - {'true_frame'}
    # "true_frame" исключаем, т.к. он не считывается а вычисляется потом

    # Формируем словарь с извлекаемыми значениями:
    row = {column: [shape.get(column, parent.get(column,
                                                 df_default_vals[column]))]
           for column in columns}
    row['track_id'] = track_id

    # Добавляем строку к датафрейму, если он был задан:
    df = pd.DataFrame(row) if df is None else \
        pd.concat([df, pd.DataFrame(row)])

    # Если остались неиспользованные поля (кроме 'shapes'), то выводим ошибку,
    # т.к. это надо проверить вручную:
    unused_params = set(shape.keys()).union(parent.keys()) - \
        set(df_columns_type.keys()) - {'shapes'}
    if len(unused_params):
        raise KeyError('Остались следующие неиспользованные поля: ' +
                       str(unused_params))

    return df


# Столбцы датафрейма для ...
# ... трека вцелом (без учёта кадров):
per_track_columns = ['group', 'source', 'attributes', 'elements', 'label']
# ... трека в отдельном кадре:
per_frame_columns = ['type', 'occluded', 'outside', 'z_order', 'rotation',
                     'points', 'frame', 'attributes']
# ... шейпа:
per_shape_columns = per_track_columns + per_frame_columns
# Всё это используется в df2annotations.

def df2annotations(df):
    '''
    Формирует из датафрейма словарь разметки в формате annotations.json.
    '''
    # Разделяем датафрейм на формы и треки:
    shapes_mask = df['track_id'].isna()
    shapes_df = df[shapes_mask]
    tracks_df = df[~shapes_mask]

    # Перебираем каждую форму:
    shapes = []
    for dfrow in shapes_df.iloc:
        shapes.append({name: dfrow[name] for name in per_shape_columns})

    # Перебираем по отдельности каждый трек:
    tracks = []
    for track_id in tracks_df['track_id'].unique():
        track_df = tracks_df[tracks_df['track_id'] == track_id]

        # Иницируем словарь, описывающий трек:
        track = {'frame': track_df['frame'].min()}
        # Требует указания первого кадра.

        # Описываем параметры, характеризующие трек в целом:
        for name in per_track_columns:

            # Записи в столбцах per_track_columns должны быть одинаковы в
            # пределах одного трека:
            vals = track_df[name]
            if df_default_vals[name] == []:  # Список переводим в кортеж,
                vals = vals.apply(tuple)     # чтобы применялась ф-ия unique
            vals = vals.unique()
            if len(vals) > 1:
                raise ValueError('Противоречивые значения в столбце '
                                 f'{name} трека {track_id}: {vals}!')
            val = vals[0]
            if isinstance(val, tuple):  # Возвращаем список, если надо
                val = list(val)
            track[name] = val

        # Дополняем покадровой разметкой:
        track_shapes = []
        for dfrow in track_df.iloc:
            shape = {name: dfrow[name] for name in per_frame_columns}
            track_shapes.append(shape)
        track['shapes'] = track_shapes

        # Вносим полное описание очередного трека в общий список:
        tracks.append(track)

    # Собираем и возвращаем итоговый словарь:
    return {'version': 0,
            'tags': [],
            'shapes': shapes,
            'tracks': tracks}


def cvat_backup_task_dir2task(task_dir, also_return_meta=False):
    '''
    Извлекает данные из подпапки с задачей в бекапе CVAT.
    Возвращает распарсенную задачу в виде списка из
    одного или более кортежей из трёх объектов:
        * DataFrame с разметкой,
        * адрес видео/фотофайла и вектора,
        * вектор номеров кадров в непрореженной последовательности.
    '''
    # Пропускаем, эсли это не папка:
    if not os.path.isdir(task_dir):
        # print(f'Пропущен "{task_dir}"!')
        return

    # Путь к папке 'data' в текущей подпапке:
    # data_dir = os.path.join(task_dir, 'data')

    # Парсим нужные json-ы task и annotations:
    with open(os.path.join(task_dir, 'task.json'),
              'r', encoding='utf-8') as f:
        task_desc = json.load(f)  # Загружаем основную инфу о видео
    with open(os.path.join(task_dir, 'annotations.json'),
              'r', encoding='utf-8') as f:
        annotations = json.load(f)  # Загружаем файл разметки

    # Загружаем данные об исходных файлах:

    # Если есть manifest.jsonl, то список файлов читаем из него:
    file = ()
    if os.path.isfile(
        manifest_file := os.path.join(task_dir, 'data', 'manifest.jsonl')
    ):

        # Читаем json-файл:
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = [json.loads(line) for line in f]

        # Формируем кортеж имён файлов:
        file = [os.path.join(task_dir, 'data', d['name'] + d['extension']) \
                for d in manifest if 'name' in d]

    # Если manifest.jsonl не существует, или файлы в нём не описаны:
    if len(file) == 0:

        # Формируем список файлов в папке data, являющихся либо изображениями,
        # либо видео:
        file = get_file_list(os.path.join(task_dir, 'data'), cv2_exts, False)
        file = sorted(file)

        # Его и берём, формируя полный путь к файлу:
        if len(file) == 1:
            file = os.path.join(task_dir, 'data', file.pop())

        '''
        # Множество должно содержать лишь один элемент:
        else:
            raise ValueError(
                f'Найдено более одного размеченного файла: {file}!')
        '''

    task_name = task_desc['name']                # Имя задачи
    jobs      = task_desc['jobs']                # Список подзадач
    data      = task_desc['data']                # Данные текущей задачи
    start = int(data['start_frame' ]           ) # Номер    первого кадра
    stop  = int(data[ 'stop_frame' ]           ) # Номер последнего кадра
    step  = data.get('frame_filter', 'filter=1') # Шаг прореживания кадров
    step  = int(step.split('=')[-1]            ) # Берём число после знака "="

    # Номера используемых кадров:
    true_frames = np.arange(start, stop + 1, step)
    true_frames = {key:val for key, val in enumerate(true_frames)}

    # Подготавливаем список размеченных фрагментов последовательности для
    # заполнения:
    task = []

    # Если список лишь из одного элемента, то берём его вместо списка:
    if len(file) == 1:
        file = file[0]

    # Перебор описаний:
    for job, annotation in zip(jobs, annotations):

        # Инициируем список датафреймов для каждой метки перед цилками чтения
        # данных:
        dfs = [new_df()]

        # Пополняем список всеми формами текущего описания:
        dfs += [shape2df(shape) for shape in annotation['shapes']]

        # Перебор объектов в текущем описании:
        for track_id, track in enumerate(annotation['tracks']):

            # Пополняем список сегментами текущего объекта для разных кадров:
            dfs += [shape2df(shape, track, track_id)
                    for shape in track['shapes']]

        # Объединяем все датафреймы в один:
        df = pd.concat(dfs)
        # Процесс объединения вынесен из цикла, т.к. он очень затратен.

        # Добавление столбца с номерами кадров полного видео (не прореженного):
        df['true_frame'] = df['frame'].apply(lambda x: true_frames[x])

        # Формируем словарь кадров для текущего фрагментов:
        start_frame = job['start_frame'] # Номер  первого   кадра текущего фрагмента
        stop_frame  = job['stop_frame']  # Номер последнего кадра текущего фрагмента
        status      = job['status']      # Статус                 текущего фрагмента
        cur_true_frames = {frame: true_frames[frame] for frame in range(start_frame, stop_frame + 1)}

        # Дополняем списки новым фрагментом данных:
        task.append((df, file, cur_true_frames))

    # Если метаданные тоже требуется вернуть:
    if also_return_meta:
        return task, task_desc

    return task


def cvat_backups2raw_tasks(unzipped_cvat_backups_dir, desc=None):
    '''
    Формирует список из распаршенных задач из папки с распакованными версиями
    CVAT-бекапов. Постаброботка вроде интерполяции контуров и разбиения на
    сцены не включена, т.е. задачи сырые.
    '''
    # Список всех дирректорий для парсинга:
    task_dirs = []

    # Список папок с распакованными датасетами:
    unzipped_cvat_backups_dirs = os.listdir(unzipped_cvat_backups_dir)

    # Перебор по всем распакованным датасетам:
    for ds_ind, cvat_ds_dir in enumerate(unzipped_cvat_backups_dirs):

        # Уточняем путь до распакованных датасетов:
        cvat_ds_dir = os.path.join(unzipped_cvat_backups_dir, cvat_ds_dir)

        # Пропускаем, если это не папка:
        if not os.path.isdir(cvat_ds_dir):
            continue

        # Название датасета:
        # cvat_ds_name = os.path.basename(cvat_ds_dir)

        # Перебор всех подпапок внутри датасета:
        for task_dir in os.listdir(cvat_ds_dir):

            # Путь к текущей подпапке:
            task_dir = os.path.join(cvat_ds_dir, task_dir)

            # Добавляем в список папок на обработку:
            task_dirs.append(task_dir)

    # Параллельная обработка данных:
    tasks = mpmap(cvat_backup_task_dir2task, task_dirs,
                  desc=desc)

    # Выбрасываем пустые задачи:
    tasks = [task for task in tasks if task]

    return tasks


def get_related_files(file, images_only=False, as_manifest=False):
    '''
    Формирует словарь перехода 
    имя_файла_изображения -> кортеж_имён_связанных_файлов или список,
    использующийся в manifest.jsonl

    В CVAT можно создавать задачи, где каждому изображению ставится в
    соответствие один и более файлов (не только изображений).
    Подробнее об этих возможностях здесь:
    https://docs.cvat.ai/docs/manual/advanced/contextual-images/
    '''

    # Из единственного файла всё равно делаем список:
    files = [file] if isinstance(file, str) else file

    resources = [] if as_manifest else {}

    # Перебираем все переданные файлы:
    for file in files:

        # Относительный и полный версии пути до его ресурсов:
        task_data_path, name, ext = split_dir_name_ext(file)
        rel_dir = os.path.join('related_images', f'{name}_{ext[1:]}')
        full_path = os.path.join(task_data_path, rel_dir)

        # Составляем список связанных с текущим файлом ресурсов:
        if os.path.isdir(full_path):

            # Получаем список всех ресурсов:
            cur_resources = get_file_list(full_path,
                                          cv2_img_exts if images_only else [])

        else:
            cur_resources = []

        # Если нужно формировать именно список для manifest.jsonl:
        if as_manifest:

            # Делаем пути относительными:
            cur_resources = [os.path.relpath(resource, task_data_path)
                             for resource in cur_resources]

            resources.append({'name': str(name),
                              'extension': ext,
                              'meta': {'related_images': cur_resources}})

        # Если нужен обычный словарь:
        else:
            resources[file] = cur_resources

    return resources


def cvat_backup_task_dir2info(task_dir):
    '''
    Определяет имя задачи и проекта по папке с задачей
    '''
    # Если передан список файлов, то берём первый:
    if isinstance(task_dir, (list, tuple)):
        task_dir = task_dir[0]
    task_dir = os.path.abspath(task_dir)

    # Если передан файл, то берём путь до его дирректории:
    if os.path.isfile(task_dir):
        task_dir = os.path.abspath(os.path.split(task_dir)[0])

    # Если указана папка с данными, берём путь до папки уровнем выше:
    if os.path.basename(task_dir) == 'data':
        task_dir = os.path.split(task_dir)[0]

    # Папка уровнем выше должна иметь имя проекта:
    proj_dir = os.path.abspath(os.path.join(task_dir, '..'))
    proj_name = os.path.basename(proj_dir)

    # Отбрасываем стандартный префикс бекапа, если есть:
    if proj_name[:8].lower() == 'project_':
        proj_name = proj_name[8:]

    # Отбрасываем стандартный cуффикс бекапа с датой, если есть:
    if proj_name[-27:-19].lower() == '_backup_':
        proj_name = proj_name[:-27]

    # Определяем имя задачи:
    with open(os.path.join(task_dir, 'task.json'),
              'r', encoding='utf-8') as f:
        task_desc = json.load(f)
    task_name = task_desc['name']
    task_subset = task_desc['subset']
    task_status = task_desc['status']

    return {'proj_name': proj_name,
            'task_name': task_name,
            'task_subset': task_subset,
            'task_status': task_status}


# Меняет список на кортеж, если возможно:
def list2tuple(value):
    if isinstance(value, list):
        return tuple(value)
    return value
# Используется в df_list2tuple.


# Меняет кортеж на список, если возможно:
def tuple2list(value):
    if isinstance(value, tuple):
        return list(value)
    return value
# Используется в df_tuple2list.


"""
def df_list2tuple(df):
    '''
    Переводит все ячейки датафрейма со списками в ячейки с кортежами.
    Используется для хеширования данных.
    '''
    return df.apply(list2tuple)

def df_tuple2list(df):
    '''
    Переводит все ячейки датафрейма со кортежами в ячейки с списками.
    Используется для восстановления данных после хеширования.
    '''
    return df.apply(tuple2list)
"""


def apply_func2df_columns(df, func, columns=None):
    '''
    Применяет заданную функцию к указанным столбцам датафрейма.
    '''
    # Если датафрейм пуст, то возвращаем его без изменений:
    if len(df) == 0:
        return df

    # Если столбцы не указаны, то обрабатываем всё:
    if columns in ['all', None]:
        df = df.apply(func)
        return df

    # Если указана строка, считаем её именем столбца и делаем из неё список
    # из одного элемента:
    if isinstance(columns, str):
        columns = [columns]

    if not isinstance(columns, (tuple, list, set)):
        raise ValueError(f'Неожиданный тип columns: {columns}!')

    # Применяем функцию:
    columns = set(columns) & set(df.keys())
    for column in columns:
        df[column] = df[column].apply(func)

    return df


def df_list2tuple(df, columns=['points', 'attributes', 'elements']):
    '''
    Переводит все ячейки датафрейма со списками в ячейки с кортежами.
    Используется для хеширования данных.
    '''
    return apply_func2df_columns(df, list2tuple, columns)


def df_tuple2list(df, columns=['points', 'attributes', 'elements']):
    '''
    Переводит все ячейки датафрейма со кортежами в ячейки с списками.
    Используется для восстановления данных после хеширования.
    '''
    return apply_func2df_columns(df, tuple2list, columns)


def df_save(df, file='df.tsv.zip'):
    '''
    Сохраняет датафрейм в файл в нужном формате.
    '''
    df = df_tuple2list(df)
    return df.to_csv(file, sep='\t', index=False)


def df_load(file='df.tsv.zip'):
    '''
    Загружает датафрейм из файла, сохранённого через df_save.
    '''
    df = pd.read_csv(file, sep='\t')
    df['points'] = df['points'].apply(eval)
    return df


def drop_label_duplicates_in_task(task):
    '''
    Исключает все повторы в разметке текущей задачи.
    '''
    # Итоговый список:
    task_ = []

    # Перебор всех подзадач:
    for df, file, true_frames in task:

        # Подготовка датафрейма для хеширования данных:
        df_ = df_list2tuple(df)

        # Исключение дубликатов строк:
        df_ = df_.drop_duplicates()

        # Если дублей не было, то оставляем подзадачу без изменений:
        if len(df_) == len(df):
            task_.append((df, file, true_frames))

        # Если дубли были, то возвращаем изменённую подзадачу:
        else:

            # Восстанавливаем датафрейм после хеширования:
            df = df_tuple2list(df)

            task_.append((df, file, true_frames))

    return task_


def drop_label_duplicates_in_tasks(tasks, desc=None):
    '''
    Удаляет повторов в разметке всего списка задач.
    '''
    # Параллельная обработка данных:
    tasks = mpmap(drop_label_duplicates_in_task, tasks,
                  desc=desc)

    return tasks


class Triangle:
    '''
    Класс, описывающий треугольник. Полезен для оценки площади и центра масс
    многоугольников.
    '''

    def __init__(self, points):
        self.points = np.array(points).reshape(3, 2)
        self._area = None
        self._center = None

    # Подсчёт площади (со знаком):
    def area(self):
        if self._area is None:
            x1, y1, x2, y2, x3, y3 = self.points.flatten()
            self._area = ((x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1)) / 2
        return self._area

    # Нахождение центра масс:
    def center(self):
        if self._center is None:
            self._center = self.points.mean(axis=0)
        return self._center

    # Общая площадь списка треугольников:
    @classmethod
    def triangles_area(cls, triangles):
        return sum([triangle.area() for triangle in triangles])

    # Общая площадь и центр масс списка треугольников:
    @classmethod
    def triangles_area_center(cls, triangles):
        area = cls.triangles_area(triangles)

        # Если площадь равна нулю, то просто центр описанного прямоугольника:
        if area == 0:
            points = np.vstack([triangle.points for triangle in triangles])
            min_points = points.min(axis=0)
            max_points = points.max(axis=0)
            return 0, (min_points + max_points) / 2
            # Этот случай полезен, чтобы не возвращать None, ведь результат
            # можно будет повторно использовать при подсчёте центра масс в
            # более общей фигуре.

        else:
            return area, sum([triangle.center() * triangle.area()
                              for triangle in triangles]) / area

    # Общий центр масс списка треугольников:
    @classmethod
    def triangles_center(cls, triangles):
        area, center = cls.triangles_area_center(triangles)
        return center

    # Разбивает многоугольник на последовательность треугольников:
    @classmethod
    def poly2triangles(cls, points, center=None):
        num_points = points.shape[0]  # Число точек контура
        assert num_points > 2         # Нужны минимум 3 точки

        # Наращиваем хвост списка точек повтором его начала:
        extended_points = np.vstack([points, points[:1, :]])
        # Так легче пробежаться по всем треугольникам, вплоть до замыкающих.

        # Если отправная точка не указана, то берётся первая в контуре:
        if center is None:
            center = points[:1, :]
            return [cls(np.vstack([center, extended_points[ind:ind + 2, :]]))
                    for ind in range(1, num_points - 1)]
        # Тогда треугольников будет на один меньше, чем точек в контуре.

        # Если отправная точка дана:
        else:
            center = np.array(center).reshape((1, 2))
            return [cls(np.vstack([center, extended_points[ind:ind + 2, :]]))
                    for ind in range(num_points)]
        # В этом случае число треугольников равно числу точек контура.

    # Подсчитывает площадь (со знаком) и центр масс многоугольника:
    @classmethod
    def poly_area_center(cls, points):
        triangles = cls.poly2triangles(points)
        return cls.triangles_area_center(triangles)

    # Подсчитывает площадь (со знаком) многоугольника:
    @classmethod
    def poly_area(cls, points):
        triangles = cls.poly2triangles(points)
        return cls.triangles_area(triangles)

    # Подсчитывает центр масс многоугольника:
    @classmethod
    def poly_center(cls, points):
        triangles = cls.poly2triangles(points)
        return cls.triangles_center(triangles)


class CVATPoints:
    '''
    Класс контуров в CVAT.
    Предоставляет набор позезных методов
    для работы с точками контура.
    '''

    def __init__(self,
                 points,
                 type_='polygon',
                 rotation=0.,
                 imsize=None,
                 rotate_immediately=True):

        '''
        if hasattr(points, 'size') and points.size == 0 or \
                len(points) == 0:
            raise ValueError('Массив точек не должен быть пустым!')
        '''

        # Переводим точки в numpy-массив:
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        # Если параметр points является вектором:
        if points.ndim == 1:

            # Сохраняем вектор в виде матрицы, где строки соответстуют точкам,
            # а столбцы - коодринатам x и y:
            self.points = points.reshape(-1, 2)

        # Если параметр points уже является матрицей:
        elif points.ndim == 2:

            # Если столбцов всего 2:
            if points.shape[1] == 2:

                # Сохраняем без изменений:
                self.points = points

            # Выводим ошибку, если столбцов не 2:
            else:
                raise ValueError(
                    f'points.shape[1] != 2; shape = {points.shape}!')

        # Выводим ошибку, если points не матрица и не вектор:
        else:
            raise ValueError('Параметр points должен быть либо вектором, ' +
                             f'либо матрицей (n, 2), а передан: {points}!')

        self.type = type_
        self.rotation = rotation  # Градусы
        self.imsize = imsize      # width, height

        # Если есть поворот и его надо применить сразу, то применяем:
        if self.rotation and rotate_immediately:
            self.points = self.apply_rot().points
            self.type = 'polygon'
            self.rotation = 0

    # Возвращает центр контура:
    def center(self):

        # Отбрасываем повторяющиеся точки, если это многоугольник:
        xy = self.fuse_multipoly() if self.type == 'polygon' else self

        if self.type in {'rectangle', 'polygon', 'polyline', 'points'}:
            # Возвращаем усреднённые значения каждой координаты:
            return self.points.mean(axis=0)
        elif self.type == 'ellipse':
            return self.points[0, :]

    # Применяет вращение к конуру на предварительно заданный угол:
    def apply_rot(self):

        if self.rotation % 360 == 0.:
            return type(self)(self.aspolygon(False).points, type='polygon',
                              imsize=self.imsize, rotate_immediately=False)

        # Получаем координаты центра контура:
        pvot = self.center()

        # Строим матрицу поворота:
        rot = np.deg2rad(self.rotation)
        sin = np.sin(rot)
        cos = np.cos(rot)
        rot_mat = np.array([[cos, sin], [-sin, cos]])

        # Принудительно переводим контур в многоугольник:
        points = self if self.type == 'polygon' else self.aspolygon(False)

        # Возвращаем контур, повёрнутый в локальной системе координат:
        return type(self)(np.matmul((points.points - pvot), rot_mat) + pvot,
                          imsize=self.imsize, rotate_immediately=False)

    # Переводит относительные YOLO-координаты в абсолютные CVAT-координаты:
    def yolo2cvat(self, height=None, width=None):

        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:

            # Если размер изображения не был задан и изначально, выводим ошибку:
            if self.imsize is None:
                raise ValueError('Должен быть задан imsize либо (height, width)!')

            height, width = self.imsize

        # Если объект - прямоугольник:
        if self.type == 'rectangle':
            xc, yc, w, h = self.flatten()

            xmin = (xc - w / 2) * width
            ymin = (yc - h / 2) * height
            xmax = (xc + w / 2) * width
            ymax = (yc + h / 2) * height

            return type(self)([xmin, ymin, xmax, ymax], 'rectangle', rotation=self.rotation, imsize=(height, width))

        # Если объект - многоугольник:
        elif self.type == 'polygon':
            x = self.x() * width
            y = self.y() * height

            return type(self)(np.vstack([x, y]).T, 'polygon', rotation=self.rotation, imsize=(height, width))

        # Если не прямоугольник, и не многоугольник:
        else:
            raise ValueError(f'Способ конвертации объекта типа "{self.type}" неизвестен!')

    # Создаёт полную копию текущего экземлпяра класса:
    def copy(self):
        return type(self)(self.points, self.type, self.rotation, self.imsize)

    # Возвращает число вершин контура:
    def __len__(self):
        return len(self.points)

    # Превращает матрицу обратно в вектор точек в формате CVAT:
    def flatten(self):
        return self.points.flatten()

    # Возвращает сдвинутый список точек:
    def shift_list(self, shift=1):
        return np.roll(self.points, shift, 0)

    # Выполняет сдвиг точек:
    def shift(self, shift):
        x0, y0 = shift
        x = self.x() + x0
        y = self.y() + y0
        return type(self)(np.vstack([x, y]).T, self.type,
                          rotation=self.rotation, imsize=self.imsize)

    # Выполняет масштабирование точек:
    def scale(self, scale):
        sx, sy = scale
        x = self.x() * sx
        y = self.y() * sy

        # Размеры изображения также меняем:
        imsize = None if self.imsize is None else (self.imsize[0] * sy, self.imsize[1] * sx)

        return type(self)(np.vstack([x, y]).T, self.type, rotation=self.rotation, imsize=imsize)

    # Возвращает координаты векторов последовательного
    # перехода от предыдущей точки контура к следующей:
    def segments_vectors(self):
        return self.shift_list(-1) - self.points

    # Возвращает длины сегментов:
    def segments_len(self):
        return np.sqrt((self.segments_vectors() ** 2).sum(1))

    # Возвращает точки в обратном порядке:
    def reverse(self):
        return self.points[::-1, :]

    # Разворачивает контур в другую сторону:
    def switch_orientation(self):
        return self.shift_list(-1)[::-1, :]

    # Возвращает абсциссы всех точек:
    def x(self):
        return self.points[:, 0]

    # Возвращает ординаты всех точек:
    def y(self):
        return self.points[:, 1]

    # Поэлементная сумма коодринат вершин двух контуров с равным количеством вершин:
    def __add__(self, cvat_points):
        return type(self)(self.points + cvat_points.points, self.type,
                          rotation=self.rotation, imsize=self.imsize)

    # Масштабирование величин контура:
    def __mul__(self, alpha):
        return type(self)(self.points * alpha, self.type,
                          rotation=self.rotation,
                          imsize=None if self.imsize is None
                          else tuple(np.array(self.imsize) * alpha))

    # Масштабирование величин контура:
    def __rmul__(self, alpha):
        return self * alpha

    # Пересечение контуров:
    def __and__(self, cvat_points):
        # Пока действует только для прямоугольников:
        assert self.type == cvat_points.type == 'rectangle'

        xmin1, ymin1, xmax1, ymax1 =        self.asbbox()
        xmin2, ymin2, xmax2, ymax2 = cvat_points.asbbox()

        # Упорядочиваем координаты по возрастанию, если очерёдность перепутана:
        if xmin1 > xmax1: xmin1, xmax1 = xmax1, xmin1
        if ymin1 > ymax1: ymin1, ymax1 = ymax1, ymin1
        if xmin2 > xmax2: xmin2, xmax2 = xmax2, xmin2
        if ymin2 > ymax2: ymin2, ymax2 = ymax2, ymin2

        # Определяем пересечение по абсциссе:
        xmin = max(xmin1, xmin2)
        xmax = min(xmax1, xmax2)

        # Если пересечения нет, возвращаем None:
        if xmin > xmax:
            return None

        # Определяем пересечение по ардинате:
        ymin = max(ymin1, ymin2)
        ymax = min(ymax1, ymax2)

        # Если пересечения нет, возвращаем None:
        if ymin > ymax:
            return None

        return type(self)([xmin, ymin, xmax, ymax], 'rectangle',
                          imsize=self.imsize)

    # Объединение контуров:
    def __or__(self, cvat_points):

        # Пока действует только для прямоугольников:
        assert self.type == cvat_points.type == 'rectangle'

        xmin1, ymin1, xmax1, ymax1 =        self.asbbox()
        xmin2, ymin2, xmax2, ymax2 = cvat_points.asbbox()

        # Определяем объединение по абсциссе:
        xmin = min(xmin1, xmin2)
        xmax = max(xmax1, xmax2)

        # Определяем пересечение по ардинате:
        ymin = min(ymin1, ymin2)
        ymax = max(ymax1, ymax2)

        return type(self)([xmin, ymin, xmax, ymax],
                          'rectangle',
                          imsize=self.imsize)

    # Перерассчитывает контур с учётом вырезания изображения:
    def crop(self, crop_bbox):

        # Создаём из параметров кропа новую рамку:
        crop_bbox = type(self)(crop_bbox,
                               'rectangle',
                               imsize=(crop_bbox[-1],
                                       crop_bbox[-2]))

        # Ищем пересечение двух прямоугольников:
        intersection = crop_bbox & self

        # Возвращаем None, если пересечений нет:
        if intersection is None:
            return None

        # Если пересечение есть, то приводим к локальной системе координат и
        # возвращаем:
        else:
            xmin, ymin, xmax, ymax = crop_bbox.flatten()
            return intersection.shift((-xmin, -ymin))

    def size(self):
        '''
        Возвращает площадь замкнутой фигуры (многоугольник, эллипс,
        прямоугольник), длину ломаной линии или число точек облака.
        Величина может быть отрицательной!
        Пока работает только для прямоугольников:
        '''
        # Если это прямоугольник - возвращаем его площадь:
        if self.type == 'rectangle':
            xmin, ymin, xmax, ymax = self.flatten()
            return (xmax - xmin) * (ymax - ymin)
        # Результат может быть отрицательным!

        # Если это набор точек - возвращаем их число:
        elif self.type == 'points':
            return len(self)

        # Если это ломаная линия - возвращаем её длину:
        elif self.type == 'polyline':
            accum = 0
            for (x1, y1), (x2, y2) in zip(self.points[:-1, :],
                                          self.points[1:, :]):
                accum += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            return accum

        # Если это эллипс - возвращаем его площадь:
        elif self.type == 'ellipse':
            cx, cy, rx, ry = self.flatten()
            return rx * ry * np.pi
        # Результат может быть отрицательным!

        # Если это многоугольник - возвращаем его площадь:
        elif self.type == 'polygon':
            return Triangle.poly_area(self.points)
        # Результат может быть отрицательным!

        else:
            raise ValueError(f'Неподдерживаемый тип: {self.type}!')

    # Неоднородное масштабирование:
    def rescale(self, k_height, k_width):
        imsize = None if self.imsize else (self.imsize[0] * k_height,
                                           self.imsize[1] * k_width)
        return type(self)(self.points * np.array([k_width, k_height]),
                          imsize=imsize)

    # Квадрат расстояния между двумя точками:
    @staticmethod
    def _distance_sqr(xy1, xy2):
        return (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2

    # Интерполяция между контурами self.points и cvat_points.points с весом
    # alpha для второго контура:
    def _morph(self, cvat_points, alpha):

        # Принудительно делаем из веса итерируемый numpy-вектор:
        a = np.array(alpha).flatten()

        # Переходим к абстрактным контурам:
        p1 = self
        p2 = cvat_points

        # Число вершин в каждом контуре
        l1 = len(p1)
        l2 = len(p2)

        # Второй коэффициент взвешинвания
        b = 1. - a

        # Делаем так, чтобы первый контур всегда был не меньше второго по числу
        # точек:
        if l2 > l1:
            p1, p2 = p2, p1
            l1, l2 = l2, l1
            a, b  = b, a

        dl = l1 - l2  # Разница числа вершин двух контуров
        # Если контуры действительно имеют разное число вершин:
        if dl:
            # Список длин каждого сегменда бОльшего контура:
            segments_len = p1.segments_len()

            # Список индексов сегментов, подлежащих схлопыванию в точку:
            short_inds = np.argsort(segments_len)[:dl]
            # Схлопываются самые короткие сегменты.

            # Список точек второго контура с повторением первой точки в конце:
            p2_circle_points = np.vstack([p2.points, p2.points[:1, :]])
            # Дублирование первой точки в конце нужно на случай, если
            # схлопывать надо последний отрезок контура, связанный с
            # первой и последней точками.

            # Переопределяем список точек второго контура, ...
            # дублируя те точки, что должны разойтись при морфинге, ...
            # образуя кратчайшие сегменты другого контура:
            points = []  # Список точек
            delay = 0    # Задержка индекса (нужен для дублирования точек)
            for ind in range(l1):  # Перебор по всем индексам бОльшего контура

                # Вносим очередную точку малого контура в список:
                points.append(p2_circle_points[ind - delay, :])

                # Если текущую точку надо продублировать, то увеличиваем
                # задержку индекса:
                if ind in short_inds:
                    delay += 1

            # Собираем точки в контур:
            p2 = type(self)(np.vstack(points), imsize=self.imsize)
        # В соответствии с вышереализованным алгоритмом в контуре с бОльшим
        # количеством точек выбираются наикратчайщие отрезки, которые будут
        # объеденины в точки при переходе в более простой многоугольник. Т.о.
        # геометрия более простого многоугольника не влияет на то, какие
        # отрезки будут вырождены в точки. При этом первые точки обоих контуров
        # обязаны переходить одна в другую. Остальные связываются в зависимости
        # от того, какие пары точек одного контура переходят в единственную
        # точку другого. Алгоритм не самый продвинутый, но кажется приемлемым в
        # данном случае.

        # Производим линейную интерполяцию:
        points = []
        for a_, b_ in zip(a, b):
            points.append(p1 * b_ + p2 * a_)

        return points[0] if len(points) == 1 else points

    # Интерполяция между контурами self.points и cvat_points.points с весом
    # alpha для второго контура:
    def morph(self, cvat_points, alpha):

        # Пока интерполируем по старому:
        return self._morph(cvat_points, alpha)

        '''
        # Интерполируем по-старому, если это возможно:
        if self.type == cvat_points.type != 'polygon':
            return self._morph(cvat_points, alpha)

        # Принудительно делаем из веса итерируемый numpy-вектор:
        a = np.array(alpha).flatten()

        # Второй коэффициент взвешинвания:
        # b = 1. - a

        # Принудительно прееоводим контуры в многоугольники:
        p1 = self.aspolygon()
        p2 = cvat_points.aspolygon()

        # Определяем число точек в каждом контуре:
        len1 = len(p1)
        len2 = len(p2)

        # Берём координаты каждой точки по отдельности:
        x1 = p1.x()
        y1 = p1.y()
        x2 = p2.x()
        y2 = p2.y()

        # Формируем матрицу квадратов расстояний:
        m = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                m[i, j] = (x1[i] - x2[j]) ** 2 + (y1[i] - y2[j]) ** 2

        # Применяем венгерский алгоритм, выполняющий оптимальные назначения:
        i, j = linear_sum_assignment(m, maximize=False)

        return i, j, self.points, cvat_points.points
        '''

    # Обрамляющий прямоугольник (левый верхний угол, правый нижний угол):
    def asrectangle(self, apply_rot=True):

        # Применяем поворот:
        if apply_rot and self.rotation % 360:
            self = self.apply_rot()

        # Получаем x- и y-координаты:
        x = self.x()
        y = self.y()

        # Вычисляем координаты рамки в зависимости от типа разметки:
        if self.type == 'ellipse':
            ax = abs(x[1] - x[0])
            ay = abs(y[1] - y[0])
            xmin = x[0] - ax
            ymin = y[0] - ay
            xmax = x[0] + ax
            ymax = y[0] + ay

        elif self.type == 'rectangle':
            xmin = x[0]
            ymin = y[0]
            xmax = x[1]
            ymax = y[1]

        elif self.type in ['polygon', 'polyline']:
            xmin = x.min()
            ymin = y.min()
            xmax = x.max()
            ymax = y.max()

        else:
            raise ValueError('Неизвестный тип сегмента: %s' % self.type)

        rect = min(xmax, xmin), min(ymax, ymin), \
            abs(xmax - xmin), abs(ymax - ymin)

        return type(self)(rect, 'rectangle', self.rotation * apply_rot,
                          imsize=self.imsize, rotate_immediately=apply_rot)
    # Параметр apply_rot пришлось ввести во избежание рекурсии при вызове
    # метода apply_rot().

    # Обрамляющий прямоугольник (левый верхний угол, размеры):
    def asbbox(self):
        return self.asrectangle().flatten()

    # Обрамляющий прямоугольник в формате YOLO (центр, размер):
    def yolobbox(self, height=None, width=None):

        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:

            # Если размер изображения не был задан и изначально, выводим
            # ошибку:
            if self.imsize is None:
                raise ValueError(
                    'Должен быть задан imsize либо (height, width)!')

            height, width = self.imsize

        # Интерпретируем параметры описанного прямоугольника как координаты
        # крайних точек:
        xmin, ymin, xmax, ymax = self.asbbox()

        cx = (xmin + xmax) / 2 / width   # Относительные
        cy = (ymin + ymax) / 2 / height  #     координаты центра
        w  = (xmax - xmin)     / width   # Относительные
        h  = (ymax - ymin)     / height  #     размеры

        return cx, cy, w, h

    # Представляет любой контур многоугольником:
    def aspolygon(self, apply_rot=True):

        if self.type == 'ellipse':
            # Параметры эллипса:
            cx, cy, rx, ry = self.flatten()
            ax = abs(rx - cx)
            ay = abs(ry - cy)

            # Точки эллипса:
            n = 64                                   # Число точек эллипса
            a = np.linspace(0, 2 * np.pi, n, False)  # Углы в радианах [0, 2pi]
            x = ax * np.cos(a) + cx
            y = ay * np.sin(a) + cy

            return type(self)(np.vstack([x, y]).T, 'polygon',
                              self.rotation * apply_rot, imsize=self.imsize,
                              rotate_immediately=apply_rot)

        elif self.type == 'rectangle':
            xmin, ymin, xmax, ymax = self.asbbox(apply_rot)
            x = np.array([xmin, xmax, xmax, xmin])
            y = np.array([ymin, ymin, ymax, ymax])

            return type(self)(np.vstack([x, y]).T, 'polygon',
                              self.rotation * apply_rot, imsize=self.imsize,
                              rotate_immediately=apply_rot)

        elif self.type == 'polygon':
            return type(self)(self.points, 'polygon',
                              self.rotation * apply_rot, imsize=self.imsize,
                              rotate_immediately=apply_rot)

        else:
            raise ValueError('Неизвестный тип сегмента: %s' % self.type)
    # Параметр apply_rot пришлось ввести во избежание рекурсии при вызове
    # метода apply_rot().

    # Конвертируем в объект другого типа (Остаётся классом CVATPoints):
    def astype(self, type_='rectangle', apply_rot=True):

        if type_ == 'polygon':
            return self.aspolygon(apply_rot, imsize=self.imsize)
        if type_ == 'rectangle':
            return type(self)(self.asbbox(), type_,
                              self.rotation * apply_rot, imsize=self.imsize)
        raise ValueError(f'Неожиданный тип: {type_}')
        # Пока ничего, кроме прямоугольника и многоугольника не
        # поддерживается.

    # Создаёт словарь аргументов для создания маски:
    def to_Mask_kwargs(self):
        return {'array': self.draw(color=255, thickness=-1).astype(bool),
                'rect': self.asbbox()}
    # Используется так:
    # mask = Mask(**points.to_Mask_kwargs())

    # Создаёт однострочный датафрейм, или добавляет новую строку к старому с
    # даннымии о контуре:
    def to_dfrow(self, df=None, **kwargs):
        return add_row2df(type=self.type,
                          points=self.flatten(),
                          rotation=self.rotation,
                          **kwargs)

    # Получить параметры для формирования cvat-разметки annotation.xml:
    def xmlparams(self):
        # Инициализация списка позиционных параметров:
        args = []
        # В настоящий момент должен содержать только тип метки.

        # Инициализация словаря именованных параметров:
        kwargs = {'rotation':
                  str(self.rotation)}

        # Тип эллипса:
        if self.type == 'ellipse':
            args.append(self.type)

            # Параметры эллипса:
            kwargs['cx'], kwargs['cy'], kwargs['rx'], kwargs['ry'] = \
                map(str, self.flatten())

        # Тип прямоугольника:
        elif self.type == 'rectangle':
            args.append('box')

            # Параметры прямоугольника:
            kwargs['xtl'], kwargs['ytl'], kwargs['xbr'], kwargs['ybr'] = \
                map(str, self.asbbox(False))

        # Тип многоугольника:
        elif self.type in {'polygon', 'polyline', 'points'}:
            args.append(self.type)

            # Параметры многоугольника:
            kwargs = {}
            kwargs['points'] = ';'.join(['%f,%f' % tuple(point)
                                         for point in self.points])

        else:
            raise ValueError('Неизвестный тип сегмента: %s' % self.type)

        return args, kwargs

    # Выполняет отражение по вертикали или горизонтали:
    def flip(self, axis={0, 1}, height=None, width=None):

        # Проверка на корректность значения axis:
        if not isinstance(axis, (list, tuple, set)):
            assert axis in [0, 1]
            axis = set((axis,))
        assert axis <= {0, 1}

        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:

            # Если размер изображения не был задан и изначально, выводим
            # ошибку:
            if self.imsize is None:
                raise ValueError(
                    'Должен быть задан imsize либо (height, width)!')

            height, width = self.imsize

        # Вычисляем новый угол поворота:
        rotation = self.rotation * (-1) ** len(axis)
        # Знак меняется на противоположный столько раз, сколько происходит
        # отражений.

        # Вычисляем новые координаты в зависимости от типа разметки:

        if self.type == 'ellipse':
            cx, cy, rx, ry = self.flatten()

            if 0 in axis:
                cy, ry = height - cy, height - ry
            if 1 in axis:
                pass
                cx, rx = width - cx, width - rx

            return type(self)([cx, cy, rx, ry], self.type, rotation=rotation,
                              imsize=(height, width), rotate_immediately=False)

        elif self.type == 'rectangle':
            xmin, ymin, xmax, ymax = self.flatten()

            if 0 in axis:
                ymin, ymax = height - ymax, height - ymin
            if 1 in axis:
                xmin, xmax = width - xmax, width - xmin

            return type(self)([xmin, ymin, xmax, ymax], self.type,
                              rotation=rotation, imsize=(height, width),
                              rotate_immediately=False)

        elif self.type == 'polygon':
            points = self.points.copy()
            if 0 in axis:
                points[:, 1] = height - points[:, 1]
            if 1 in axis:
                points[:, 0] = width - points[:, 0]

            return type(self)(points, self.type, rotation=rotation,
                              imsize=(height, width), rotate_immediately=False)

        else:
            raise ValueError('Неизвестный тип сегмента: %s' % self.type)

    # Выполняет отражение по горизонтали:
    def fliplr(self, height=None, width=None):
        return self.flip(1, height, width)

    # Выполняет отражение по вертикали:
    def flipud(self, height=None, width=None):
        return self.flip(0, height, width)

    # Поворот k-раз на 90 градусов по часовой стрелке:
    def rot90(self, k=1, height=None, width=None):

        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:

            # Если размер изображения не был задан и изначально, выводим
            # ошибку:
            if self.imsize is None:
                raise ValueError(
                    'Должен быть задан imsize либо (height, width)!')

            height, width = self.imsize

        # Приводим к диапазону [0, 4) (т.е. поворот от 0 до 270 градусов):
        k = k - int(np.ceil((k + 1) / 4) - 1) * 4
        assert k in {0, 1, 2, 3}

        # Если 0 градусов, то просто повторяем контур:
        if k == 0:
            return type(self)(self.points, self.type, rotation=self.rotation,
                              imsize=(height, width), rotate_immediately=False)

        # Если 180 градусов, то вместо поворота выполняем отражения по
        # горизонтали и вертикали:
        elif k == 2:
            return self.flip(height=height, width=width)

        # Если 90 или 270 градусов:
        else:  # (k in {1, 3})

            # Меняем местами ширину и высоту итогового изображнеия:
            imsize = (width, height)

            if self.type == 'ellipse':
                cx, cy, rx, ry = self.flatten()

                if k == 1:
                    cx, cy, rx, ry = cy, width - cx, ry, width - rx
                else:
                    cx, cy, rx, ry = height - cy, cx, height - ry, rx

                return type(self)([cx, cy, rx, ry], self.type,
                                  rotation=self.rotation, imsize=imsize,
                                  rotate_immediately=False)

            elif self.type == 'rectangle':
                xmin, ymin, xmax, ymax = self.flatten()

                if k == 1:
                    xmin, ymin, xmax, ymax = ymax, width - xmin, ymin, width - xmax
                else:
                    xmin, ymin, xmax, ymax = height - ymax, xmin, height - ymin, xmax

                return type(self)([xmin, ymin, xmax, ymax], self.type,
                                  rotation=self.rotation, imsize=imsize,
                                  rotate_immediately=False)

            elif self.type == 'polygon':
                points = self.points[:, ::-1].copy()

                if k == 1:
                    points[:, 1] = width - points[:, 1]
                else:
                    points[:, 0] = height - points[:, 0]

                return type(self)(points, self.type, rotation=self.rotation,
                                  imsize=imsize, rotate_immediately=False)

    # Номера точек, где контур надо резать на два:
    def get_split_inds(self):

        # Если фигура - многоугольник, содержащий более 3х точек:
        if self.type == 'polygon' and len(self) > 3:

            # Формируем список индексов двух одинаковых точек, идущих в
            # контуре подрят:
            split_inds = np.where((
                self.points == self.shift_list()).all(1))[0]
            # Ищем пересечение множеств индексов, от которых идёт совпадение
            # значений со сдвигом на 1.

            # Определяем, есть ли в контуре три одинаковые точки, идущие
            # подрят:
            inds2del = set(split_inds) & \
                set(np.where((self.points == self.shift_list(2)).all(1))[0])
            # Ищем пересечение множеств индексов, от которых идёт совпадение
            # значений со сдвигом на 1 и 2

            # Если есть:
            if inds2del:

                # Если в контуре есть 4 и более одинаковых точек, идущих
                # подрят, то у нас проблемы:
                if inds2del & set(np.where((
                        self.points == self.shift_list(3)).all(1))[0]):
                    raise IndexError(
                        'В контуре совпадают более 3х точек подряд!\n' +
                        'Этот контур уже нельзя разбить автоматически!')

                # Отбрасываем ненужные индексы:
                split_inds = np.array(
                    sorted(list(set(split_inds) - set(inds2del))))

            return split_inds

        # Если фигура - не мгогоугольник, то возвращаем пустой список:
        else:
            return []

    # Расщепляет контур, если в нём на самом деле сохранены несколько контуров:
    def split_multipoly(self):

        # Копируем текущий объект:
        copy = self.copy()
        # Нужно для применения поворота.

        # Расщепляем только если это многоугольник:
        if copy.type != 'polygon':
            return [copy]

        # Применяем поворот, если надо:
        if copy.rotation:
            copy.points = copy.apply_rot().points
            copy.rotation = 0

        # Получаем точки расщепления:
        split_inds = copy.get_split_inds()

        # Если точки вообще имеются, то:
        if len(split_inds):

            cvat_points_list = []
            for start, end in zip(np.roll(split_inds, 1) + 1, split_inds):
                lenght = end - start
                if lenght < 0:
                    lenght += len(copy)

                sub_points = copy.shift_list(-start)[:lenght, :]
                sub_points = type(copy)(
                    sub_points, rotation=copy.rotation, imsize=copy.imsize,
                    rotate_immediately=True)
                cvat_points_list.append(sub_points)

            return cvat_points_list

        else:
            return [copy]

    # Собирает один контур из нескольких:
    @classmethod
    def unite_multipoly(cls, poly_list, rotation=0., imsize=None):

        # Если контуров реально несколько, то действительно объединяем:
        if len(poly_list) > 1:

            # Создаём копию точек первого контура:
            points = cls(poly_list[0]).fuse_multipoly().points
            # При этом на всякий случай удаляем повторяющиеся точки.

            # Берём последную точку этого контура:
            last_point = points[-1:, :]

            # Последнюю точку дублируем в самом контуре, чтобы отметить место
            # начала следующего контура:
            points = np.vstack([points, last_point])

            # Наращиваем контур, дублируя в каждой из составляющих последние
            # точки:
            for new_poly in poly_list[1:]:
                new_points = cls(new_poly).fuse_multipoly().points
                points = np.vstack([points, new_points, new_points[-1:, :]])
            # В каждой составляющей также удаляем повторяющиеся точки.

            try:
                cls(points, rotation=rotation, imsize=imsize).split_multipoly()
            except:
                print(poly_list)
                print(points)
                raise

            return cls(points, rotation=rotation, imsize=imsize)

        # Если контур всего один, то просто делаем его копию:
        elif len(poly_list) == 1:
            return cls(poly_list[0], rotation=rotation, imsize=imsize)

        # Если контуров вообще нет, то возвращаем None:
        else:
            return
    # Все контуры объединяются в один, а места сращивания отмечаются
    # дублированием координат последней точки предыдущего контура, чтобы
    # потом можно было без проблем восстановить исходный набор конуров.

    # Таблица квадратов расстояний между каждой парой точек из разных
    # контуров:
    @classmethod
    def _distance_sqr_matrix(cls, points1, points2=None):
        return apply_on_cartesian_product(cls._distance_sqr,
                                          points1, points2, True, 0,
                                          num_procs=1)

    # Находит ближайшую пару точек:
    @classmethod
    def _nearest_points(cls, points1, points2=None):

        # Формируем матрицу расстояний для всех пар:
        distance_sqr_matrix = cls._distance_sqr_matrix(points1, points2)

        # Инициируем кротчайшее расстояние:
        dist_ = np.inf

        # Если пары перебираются в пределах одного контура:
        if points2 is None:
            for ind1 in range(distance_sqr_matrix.shape[0] - 1):
                for ind2 in range(1, distance_sqr_matrix.shape[1]):
                    dist = distance_sqr_matrix[ind1, ind2]
                    if dist < dist_:
                        ind1_, ind2_, dist_ = ind1, ind2, dist
            # Спаривание точек с самими собой исключаем.

        # Если пары выбираются из двух контуров:
        else:
            for ind1 in range(distance_sqr_matrix.shape[0]):
                for ind2 in range(distance_sqr_matrix.shape[1]):
                    dist = distance_sqr_matrix[ind1, ind2]
                    if dist < dist_:
                        ind1_, ind2_, dist_ = ind1, ind2, dist

        assert distance_sqr_matrix.min() == dist_
        assert distance_sqr_matrix[ind1_, ind2_] == dist_
        return ind1_, ind2_, dist_

    # Отбрасывает метки мультиконтурности (убирает повторяющиеся точки):
    def smart_fuse_multipoly(self):

        # Немногоугольники возвращаем без изменений:
        if self.type != 'polygon':
            return self.copy()

        # Расщипляем на отдельные контуры:
        points_list = self.split_multipoly()

        # Возвращаем None, если контуров нет:
        if points_list is None:
            return None

        # Если контур лишь один, то его и возвращаем:
        elif len(points_list) == 1:
            return points_list[0]

        ################################################
        # Заносим всевозможные "перемычки" в dists_df: #
        ################################################

        # Инициируем датафрейм связности частей сегмента:
        dists_df = pd.DataFrame(columns=['pts_ind1',
                                         'pts_ind2',
                                         'distance',
                                         'pt1_ind',
                                         'pt2_ind',
                                         'status'])

        # Заполняем датафрейм связности частей сегмента:
        min_dist = np.inf
        for pts_ind2, p2 in enumerate(points_list[1:], 1):
            for pts_ind1, p1 in enumerate(points_list[:pts_ind2]):

                # Находим минимальное расстояние между точками
                # двух контуров:
                pt1_ind, pt2_ind, dist = self._nearest_points(
                    p1.points, p2.points)

                # Если текущая пара является ближайшей из всех
                # рассмотренных, то запоминаем её индекс:
                if min_dist > dist:
                    nearest_ind = len(dists_df)
                    min_dist = dist

                    # Множество индексов задействованных контуров:
                    used_pts_inds = {pts_ind1, pts_ind2}

                # Заносим данные в таблицу:
                dists_df.loc[len(dists_df), :] = [pts_ind1,
                                                  pts_ind2,
                                                  dist,
                                                  pt1_ind,
                                                  pt2_ind,
                                                  'unused']

        ##############################################
        # Ищем оптимальное подмножество "перемычек": #
        ##############################################

        # Меняем статус для пары ближайших сегментов:
        dists_df.loc[nearest_ind, 'status'] = 'used'

        # Множество индексов всех контуров:
        all_pts_inds = set(range(len(points_list)))

        # Множество индексов НЕзадействованных контуров:
        unused_pts_inds = all_pts_inds - used_pts_inds

        # Пока есть незадействованные контуров:
        while unused_pts_inds:

            # Маскирование пар задействованных контуров с незадействованными:
            df_mask = dists_df['status'] == 'unused'
            # Формируем и применяем маски для задействованных контуров:
            submask = dists_df['status'] == 'used'
            for pts_ind in used_pts_inds:
                submask |= dists_df['pts_ind1'] == pts_ind
                submask |= dists_df['pts_ind2'] == pts_ind
            df_mask &= submask
            # Формируем и применяем маски для незадействованных контуров:
            submask = dists_df['status'] == 'used'
            for pts_ind in unused_pts_inds:
                submask |= dists_df['pts_ind1'] == pts_ind
                submask |= dists_df['pts_ind2'] == pts_ind
            df_mask &= submask
            # Т.е. мы строим следующие рёбра от уже задействованных вершин
            # графа.

            # Получаем индекс маскированной пары с минимальным расстоянием:
            used_pts_ind = dists_df.loc[df_mask, 'distance'].idxmin()

            # Отмечаем в датафрейме найденную пару как задействованную:
            dists_df.loc[used_pts_ind, 'status'] = 'used'

            # Переносим новый контур из множества незадействованных в
            # задействованные:
            new_used_pts_ids = set(dists_df.loc[used_pts_ind, ['pts_ind1',
                                                               'pts_ind2']])
            unused_pts_inds -= new_used_pts_ids
            used_pts_inds |= new_used_pts_ids

        # Формируем словарь перемычек:
        bridges = {}
        for dists_dfrow in dists_df.iloc:
            # Используем только задействованные перемычки:
            if dists_dfrow['status'] == 'unused':
                continue

            key1 = (dists_dfrow['pts_ind1'], dists_dfrow['pt1_ind'])
            key2 = (dists_dfrow['pts_ind2'], dists_dfrow['pt2_ind'])
            extend_list_in_dict_value(bridges, key1, [key2])
            extend_list_in_dict_value(bridges, key2, [key1])

        ##################################
        # Строим контур с "перемычками": #
        ##################################

        # Инициируем начальное состояние индекса текущего контура и индекса
        # его текущей точки:
        cur_pts_ind = 0
        cur_pt_ind = 0
        cur_pt_ind = CircleInd(len(points_list[cur_pts_ind]), cur_pt_ind)

        # Длина итогового контура:
        total_points = sum(map(len, points_list)) + \
                       sum(map(len, bridges.values()))
        # Состоит из совокупного числа точек составных контуров и
        # удвоенного числа перемычек (по одному в каждую сторону).

        points = []  # Инициируем итоговый контур

        # Инициируем множество начатых контуров
        # (вдоль которых был выполнен хотя бы один шаг):
        used_pts_inds = set()
        # Используется чтобы избежать возврата на предыдущий контур по
        # перемычке до того, как весь текущий контур будет добавлен.

        # Пока итоговый контур не наполнен:
        while len(points) < total_points:
            # Индекс текущей точки текущего контура:
            cur_pt = points_list[cur_pts_ind].points[int(cur_pt_ind), :]

            # Если текущий контур не занесён в список начатых, то исправляем
            # это и делаем шаг к следующей точке в текущем контуре не взирая
            # на то, есть ли в текущей точке перемычки в другие контуры, или
            # нет:
            if cur_pts_ind not in used_pts_inds:
                used_pts_inds.add(cur_pts_ind)

                # Вносим текущую точку в итоговый список и переходим к
                # следующей:
                points.append(cur_pt)
                cur_pt_ind.inc()
                continue
            # Такое поведение позволяет избегать возврата на предыдущий
            # контур, не пройдя по текущему.

            # Получаем перемычку из текущей точки с другим контуорм, если она
            # есть:
            cur_state = (cur_pts_ind, int(cur_pt_ind))
            new_state_list = bridges.get(cur_state, [])

            # Если перемычка есть:
            for new_state in new_state_list:
                new_pts_ind, new_pt_ind = new_state

                # Если перемычка ведёт в неначатый контур:
                if new_pts_ind not in used_pts_inds:

                    # Добавляем текущую точку в итоговый контур:
                    points.append(cur_pt)

                    # Переходим в новый контур:
                    cur_pts_ind, cur_pt_ind = new_pts_ind, new_pt_ind
                    cur_pt_ind = CircleInd(len(points_list[cur_pts_ind]),
                                           cur_pt_ind)

                    # Использованную перемычку удаляем из списка:
                    new_state_list.remove(new_state)

                    # Другие перемычки не рассматриваем:
                    break

            # Если перемычек в неначатый контур не нашлось:
            else:

                # Из текущего начатого контура не должно быть более одной
                # перемычки, ведущей в другой начатый контур:
                assert len(new_state_list) <= 1

                # Если есть перемычка в начатый контур, то:
                if len(new_state_list):

                    # Заносим текущую точку в итоговый контур:
                    points.append(cur_pt)

                    # Возвращаемся по перемычке назад, попутно удаляя её из
                    # списка:
                    cur_pts_ind, cur_pt_ind = new_state_list.pop()
                    cur_pt_ind = CircleInd(len(points_list[cur_pts_ind]),
                                           cur_pt_ind)

                # Если перемычек нет, то идём вдоль текущего контура:
                else:

                    # Вносим текущую точку в итоговый список и переходим к
                    # следующей:
                    points.append(cur_pt)
                    cur_pt_ind.inc()

        # Собираем из списка новый массив точек:
        return type(self)(points, imsize=self.imsize)
    # Применяется если имеется реальная многоконтурность, но надо
    # визуализировать результат в CVAT. Т.к. сам CVAT не поддерживает
    # хранение нескольких многоугольников в одном сегменте, то они будут
    # объеденины в один посредством кратчайших "перемычек".

    # Для тестирования smart_fuse_multipoly:
    '''
    imsize = (100, 100)

    while True:
        m = np.zeros(imsize, np.uint8)

        for i in range(np.random.randint(1, 7)):
            x = np.random.randint(imsize[1], size=2)
            y = np.random.randint(imsize[0], size=2)
            m[min(y):max(y), min(x):max(x)] = 255

        p = CVATPoints.from_mask(m)
        if p is not None:
            break

        print(1)

    p_ = p.smart_fuse_multipoly()

    plt.figure(figsize=(24, 24))
    plt.imshow(np.hstack([m, 255 - p_.draw(), p_.draw(thickness=-1)]))
    plt.axis(False);
    '''

    # Отбрасывает метки мультиконтурности (убирает повторяющиеся точки):
    def fuse_multipoly(self):
        # Немногоугольники возвращаем без изменений:
        if self.type != 'polygon':
            return self.copy()

        points = list(self.points)  # Формируем список точек
        points.append(points[0])
        # Добавляем в конец первую точку для проверки на повторение
        # первого элемента с последним.

        # Инициируем новый список:
        points_ = []

        # Заполняем его неповторяющимися элементами:
        for ind in range(len(self.points)):
            if (points[ind] != points[ind + 1]).any():
                points_.append(points[ind])

        # Собираем из списка новый массив точек:
        return type(self)(np.array(points_), self.type,
                          rotation=self.rotation, imsize=self.imsize,
                          rotate_immediately=False)
    # Применяется если повторяющиеся точки появились случайно, например, ...
    # ... при разметке, и не являются признаком многоконтурности.

    # Векторизирует маску:
    @classmethod
    def from_mask(cls, mask, findContours_mode=cv2.CHAIN_APPROX_SIMPLE):

        # Бинарную маску надо перевести в полутоновую:
        if str(mask.dtype) == 'bool':
            mask = mask.astype(np.uint8)

        # Векторизируем:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, findContours_mode)

        # Убираем лишнее измерение в каждом контуре и оставляем лишь те
        # контуры, что вообще содержат точки:
        contours = [contour.squeeze().astype(int)
                    for contour in contours if len(contour)]
        # Также переводим координаты в целочисленный тип.

        # Собираем контуры в один:
        points = cls.unite_multipoly(contours, imsize=tuple(mask.shape[:2]))

        return points

    # Создаёт прямоугольный контур из обрамляющего прямоугольника:
    @classmethod
    def from_bbox(cls, xmin, ymin, dx, dy, imsize=None):

        xmax = xmin + dx
        ymax = ymin + dy

        return cls([xmin, ymin, xmax, ymax], 'rectangle', imsize=imsize)

    # Создаёт прямоугольный контур из обрамляющего прямоугольника YOLO-формата:
    @classmethod
    def from_yolobbox(cls, cx, cy, w, h, imsize):

        imheight, imwidth = imsize

        cx = cx * imwidth
        cy = cy * imheight
        w  = w  * imwidth
        h  = h  * imheight

        xmin = cx - w / 2
        ymin = cy - h / 2

        xmax = xmin + w
        ymax = ymin + h

        return cls([xmin, ymin, xmax, ymax], 'rectangle', imsize=imsize)

    # Создаёт многоугольник из сегмента YOLO-формата:
    @classmethod
    def from_yoloseg(cls, points, imsize=None):

        # Приводим множество точек в нужный формат:
        points = np.array(points).reshape(-1, 2)

        # Переводим относительные координаты в абсолютные, если указан размер
        # изображения:
        if imsize is not None:
            points[:, 0] *= imsize[1]  # X
            points[:, 1] *= imsize[0]  # Y

        return cls(points, 'polygon', imsize=imsize)

    # Создаёт контур из строки в датафрейме подзадачи:
    @classmethod
    def from_dfrow(cls, raw, imsize=None, rotate_immediately=True):
        return cls(raw['points'], raw['type'], raw['rotation'], imsize=imsize,
                   rotate_immediately=rotate_immediately)

    # Возвращает многоугольный контур с уменьшенным числом вершин:
    def reducepoly(self, epsilon=1.5):
        assert self.type == 'polygon'

        # Возвращаем список без изменений, если он пустой:
        if len(self.points) == 0:
            return self.copy()

        # Получаем Расщеплённые контуры:
        multipoly = self.split_multipoly()

        # Если контуров больше одного:
        if len(multipoly) > 1:

            # Обрабатываем каждый по тдельности:
            multipoly = [poly.reducepoly(epsilon).points for poly in multipoly]

            # Объединяем их обратно в один контур и возвращаем:
            return self.unite_multipoly(multipoly, rotation=self.rotation,
                                        imsize=self.imsize)

        # Если контур всего один, то обрабатываем его:
        reduced_poly = cv2.approxPolyDP(self.points.astype(int),
                                        epsilon, True).squeeze()

        # Если в упрощённом контуре меньше 3-х точек, то возвращаем исходный
        # контур:
        if reduced_poly.size < 6:
            return self.copy()

        # Возвращаем упрощённый контур:
        return type(self)(reduced_poly, rotation=self.rotation,
                          imsize=self.imsize, rotate_immediately=False)

    # Многоугольник в формате YOLO:
    def yoloseg(self, height=None, width=None):

        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:

            # Если размер изображения не был задан и изначально, выводим
            # ошибку:
            if self.imsize is None:
                raise ValueError(
                    'Должен быть задан imsize либо (height, width)!'
                )

            height, width = self.imsize

        # Конвертируем точки в многоугольник, если нужно:
        points = self if self.type == 'polygon' else self.aspolygon()

        # Интерпретируем параметры описанного прямоугольника как координаты
        # крайних точек:
        x = points.x() / width   # Относительная абсцисса
        y = points.y() / height  # Относительная ордината

        return np.vstack([x, y]).T.flatten()

    # Рисует метку на изображении:
    def draw(self,
             img=None,
             caption=None,
             color=(255, 255, 255),
             thickness=1,
             alpha=1.,
             show_start_point=False):

        # Поворачиваем контур, если объект не эллипс, а угол кратен 360
        # градусам:
        if self.type != 'ellipse' and self.rotation % 360:
            self = self.apply_rot()

        # Если изображение не задано, то:
        if img is None:

            # Если размер изображения не записан и в самом контуре, то
            # используем размер самого многугольника:
            if self.imsize is None:
                shift = self.points.min(0)                              # Определяем координаты левого верхнего края обрамляющего прямоугольника
                points = self.points - shift                            # Прижимаем многоугольник к левому верхнему углу
                points = points.astype(int)                             # Округляем координаты вершин до целых
                img = np.zeros(points.max(0)[::-1] + 1, dtype=np.uint8) # Размер изображения = размеру многоугольника
            else:
                img = np.zeros(self.imsize, dtype=np.uint8)
                shift = np.zeros(2)

        # Если изображение задано, то контур берём как есть и округляем
        # коодринаты до целых:
        else:
            img = img.copy()
            shift = np.zeros(2)

        # Если отрисовка с полупрозрачностью, то создаём копию исходного
        # изображения для последующего смешения:
        if alpha < 1:
            org_img = img.copy()

        # Отрисовываем многоугольник:
        if self.type in ['polygon', 'polyline']:

            # Расщепляем составные контуры:
            multipoly = self.shift(-shift).split_multipoly()

            # Формируем списки списокв точек:
            pts = [p.points.astype(int) for p in multipoly]

            # Рисуем залитый или полый контур:
            if (self.type == 'polygon') and (thickness < 0):
                img = cv2.fillPoly(img, pts, color)
            elif thickness > 0:
                img = cv2.polylines(img, pts, self.type == 'polygon',
                                    color=color, thickness=thickness)

            # Определение координат центра надписи:
            if caption:
                cx = int(self.x().mean())
                cy = int(self.y().mean())

            # Обводим кружком первую вершину контура, если надо:
            if show_start_point:
                for p in multipoly:
                    img = cv2.circle(img, p.points[0, :].astype(int), 20,
                                     color=color, thickness=thickness)

        # Отрисовываем прямоугольник:
        elif self.type == 'rectangle':
            xmin, ymin, xmax, ymax = self.flatten().astype(int)
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                                color=color, thickness=thickness)
            if caption:
                cx = int(self.x().mean())
                cy = int(self.y().mean())

        # Отрисовываем эллипс:
        elif self.type == 'ellipse':
            cx, cy, rx, ry = self.flatten().astype(int)
            ax = abs(rx - cx)
            ay = abs(ry - cy)
            img = cv2.ellipse(img, (cx, cy), (ax, ay), self.rotation,
                              0, 360, color=color, thickness=thickness)

        # Отрисовываем точки:
        elif self.type == 'points':

            # Рисуем только если толщина больше нуля:
            if thickness > 0:
                for ind in range(len(self)):
                    img = cv2.circle(img, self.points[ind, :].astype(int),
                                     thickness, color=color, thickness=-1)

            # Определяем координаты центра надписи:
            if caption:
                cx = int(self.x().mean())
                cy = int(self.y().mean())

        else:
            raise ValueError(f'Неизвестный тип контура "{self.type}"!')

        # Надпись в центре, если надо:
        if caption:
            img = put_text_carefully(str(caption), img, (cy, cx), color=color)

        # Если отрисовка с полупрозрачностью, то смешиваем результат с исходным
        # изображением:
        if alpha < 1:
            img = cv2.addWeighted(img, alpha, org_img, 1 - alpha, 0)

        return img

    # Рисует многоугольник на изображении и выводит последнее на экран:
    def show(self, *args, **kwargs):
        plt.imshow(self.draw(*args, **kwargs))
        plt.axis(False)


class CVATLabels:
    '''
    Класс, хранящий все типы меток CVAT-проекта (классы)
    '''
    def __init__(self, cvat_raw_labels):

        # Если передано имя текстового файла, читаем его как json:
        if isinstance(cvat_raw_labels, str) and \
                os.path.isfile(cvat_raw_labels):
            with open(cvat_raw_labels, 'r', encoding='utf-8') as f:
                cvat_raw_labels = json.load(f)

        # Если передан список, считаем, что он и содержит словари всех меток:
        elif isinstance(cvat_raw_labels, list):
            pass

        # Иных вариантов не предусмотрено:
        else:
            raise ValueError('Неожиданное значение "cvat_raw_labels": ' +
                             f'{cvat_raw_labels}!')

        # Формируем датафрейм:
        self.df = pd.DataFrame.from_dict(cvat_raw_labels)

    # Возвращает полный список имён классов:
    def labels(self):
        return list(self.df['name'])

    # Возвращает цвет (RGB), соответствующий классу:
    def label2color(self, label):
        line = self.df[self.df['name'] == label]['color']
        assert len(line) == 1
        return ImageColor.getcolor(line.values[0], "RGB")

    # Возвращает аргументы создания радиокнопки списка классов в Jupyter:
    def get_ipy_radio_buttons_kwargs(self, description='Класс:'):
        return  {'options': self.labels(), 'description': description}
    # from IPython.display import display
    # from ipywidgets import RadioButtons
    #
    # cvat_labels = CVATLabels(...)
    # rb = RadioButtons(**cvat_labels.get_ipy_radio_buttons_kwargs())
    #
    # def on_button_clicked(b):
    #     ...
    # rb.observe(on_button_clicked, names='value')
    # display(rb)


def interpolate_df(df, true_frames):
    '''
    Дополняет датафрейм интерполированными контурами.
    '''
    # Номер последнего кадра прореженной последовательности:
    last_frame = np.max(list(true_frames.keys()))

    # Список датафреймов с интерполированными кадрами:
    interp_frame_dfs = []

    # Перебор всех объектов:
    for track_id in df['track_id'].unique():

        # Пропускаем объекты без идентификатора, ...
        # ... т.к. обычно это означает, что объект  ...
        # ... появляется лишь в одном кадре:
        if track_id is None:
            continue

        # Берём ту часть датафрейма, в которой содержится инфомрация о текущем
        # объекте:
        object_df = df[df['track_id'] == track_id]

        # Определяем последний кадр и его :
        last_key_frame =           object_df['frame'].max()              # Номер последнего ключевого кадра
        last_key       = object_df[object_df['frame'] == last_key_frame] # Датафрейм с последним ключём
        assert len(last_key) == 1                                        # Ключ должен быть только один
        #last_key_row = last_key.iloc[0]                                  # Для более лёгкой адресации к полям

        # Если последний кадр не закрывает анимацию, то копируем ключ на
        # последний кадр в оба датафрейма:
        if last_key_frame < last_frame and not last_key['outside'].iloc[0]:
            row = last_key.copy()
            row[     'frame'] =             last_frame
            row['true_frame'] = true_frames[last_frame]
            df        = pd.concat([       df, row])
            object_df = pd.concat([object_df, row])

        # Инициируем первый кадр новой последовательности с объектом:
        start_frame = None

        # Перебор по всем кадрам, где отмечен объект:
        for frame in sorted(object_df['frame'].unique()):

            # Строка, описывающая текущий объект в текущем кадре:
            frame_row = object_df[object_df['frame'] == frame].iloc[0]

            # Если предыдущий ключевой кадр был, и между ним и текущим есть ещё
            # хоть один кадр:
            if (start_frame is not None) and (frame > start_frame + 1):

                # Строка, описывающая текущий объект в предыдущем ключевом
                # кадре:
                sart_frame = object_df[object_df['frame'] == start_frame]
                assert len(sart_frame) == 1
                sart_frame_row = sart_frame.iloc[0]

                # Контуры предыдущего и текущего ключевых кадров
                p1 = CVATPoints(sart_frame_row['points'], sart_frame_row['type'])
                p2 = CVATPoints(     frame_row['points'],      frame_row['type'])

                # Весовые коэффициенты для морфинга промежуточных кдаров:
                alphas = np.linspace(0, 1,
                                     frame - start_frame,
                                     endpoint=False)[1:]

                # Контуры для промежуточных кадров:
                points_list = p1.morph(p2, alphas)
                if isinstance(points_list, CVATPoints):
                    points_list = [points_list]

                # Добавляем промежуточные кадры:
                for interp_frame, points in enumerate(points_list,
                                                      start_frame + 1):
                    row = sart_frame.copy()
                    row[    'points'] = [list(points.flatten())]
                    row[     'frame'] =             interp_frame
                    row['true_frame'] = true_frames[interp_frame]
                    interp_frame_dfs.append(row)

            # Если на этом кадре объект скрывается, то сбрасываем start_frame,
            # иначе обновляем:
            start_frame = None if frame_row['outside'] else frame

    # Добавляем интерполированные кадры к датафрейму:
    df = pd.concat([df] + interp_frame_dfs)
    # Операция вынесена за пределы вложенных циклов ради производительности.

    # Возвращаем отсортированный по кадрам проинтерполированный список:
    return df.sort_values('frame')


def interpolate_task(task):
    '''
    Дополняет все датафреймы текущей задачи интерполированными контурами.
    '''
    return [(interpolate_df(df, true_frames), file, true_frames)
            for df, file, true_frames in task]


def interpolate_tasks_df(tasks, desc=None):
    '''
    Интерполирует контуры всех датафреймов в датасете:
    '''
    # Параллельная обработка данных:
    return mpmap(interpolate_task, tasks, desc=desc)


def split_subtask_by_labels(subtask                           ,
                            debug_mode: 'Режим отладки' = True):
    '''
    Дробит подзадачу (df, file, true_frames) на более мелкие подзадачи,
    согласно меткам "scene-start" и "scene-finish" в датафрейме.
    Применять надо строго после интерполяции контуров!
    '''
    # Расщипляем задачу на составляющие:
    full_df, file_path, true_frames = subtask

    # Определяем имя проекта и задачи для отладки:
    task_dir = os.path.dirname(file_path) if isinstance(file_path, str) \
        else os.path.dirname(file_path[0])
    task_info = cvat_backup_task_dir2info(task_dir)
    proj_name, task_name = task_info['proj_name'], task_info['task_name']

    # Строка описания файла(ов):
    file_path_str = file_path if isinstance(file_path, str) \
        else '|'.join(file_path)

    # Проверяем заполненность датафрейма:
    if full_df is None:
        debug_str = 'Не заполнен "%s"\n' % file_path_str

        if debug_mode:
            print(debug_str)

            # Возвращаем задачу без изменений, но обёрнутую в список:
            return [subtask]

        else:
            raise ValueError(debug_str)

    # Формируем датафрейм, по которому будем проводить расщепление:
    sf_df = full_df[full_df['outside'] == False]  # Убераем все скрытые сегменты
    sf_df = sf_df[(sf_df['label'] == 'scene-start' ) | \
                  (sf_df['label'] == 'scene-finish')]
    # Оставляем только метки начала/конца сцены.
    # Этот датафрейм содержит только действующие метки начала/конца сцены.

    # Если меток начала/конца сцены нет, то возвращаем подзадачу, обёрнув её в
    # список:
    if len(sf_df) == 0:
        return [subtask]

    # Формируем датафрейм, подлежащий расщеплению:
    df = full_df[(full_df['label'] != 'scene-start' ) & \
                 (full_df['label'] != 'scene-finish')]
    # Из него выброшены все строки, связанные с метками начала и конца сцены.

    # Инициируем нужные переменные перед циклом:
    debug_str   = file_path_str + ':\n' # Строка с отладочными данными
    start_frame = None                  # Номер первого кадра последовательности
    subtasks    = []                    # Список разбитых подзадач

    # Проход по всем номерам прореженных кадров, где есть метки начала/конца
    # сцены:
    for frame in sorted(sf_df['frame'].unique()):

        # Метки начала/конца сцены, касающиеся только текущего кадра:
        frame_data = sf_df[sf_df['frame'] == frame]

        # Номер текущего кадра в непрореженной последовательности:
        true_frame = frame_data['true_frame'].values[0]

        # Если в кадре есть метка начала последовательности:
        if (frame_data['label'] == 'scene-start').any():
            debug_str += '(%d ' % frame

            # Если при этом последовательность уже открыта, то выводим ошибку:
            if start_frame is not None:
                debug_str += ' <- Два начала последовательности в метках'
                debug_str += ' подряд в кадре №%d(%d)' % (frame, true_frame)
                debug_str += ' проекта "%s"' % proj_name
                debug_str += ' задачи "%s"!\n' % task_name
                if debug_mode:
                    print(debug_str)
                    break
                else:
                    raise ValueError(debug_str)

            # Если при этом последовательность ещё не открыта, то открываем:
            else:
                start_frame = frame

        # Если в кадре есть метка конца последовательности:
        if (frame_data['label'] == 'scene-finish').any():
            debug_str += ')%d ' % frame

            # Если при этом последовательность пока открыта:
            if start_frame is not None:

                # Формируем соответствующий этому интервалу датафрейм в список:
                sub_df = df[(df['frame'] >= start_frame ) & \
                            (df['frame'] <=       frame )]

                # Формируем соответствующий этому интервалу словарь номеров
                # кадров:
                sub_true_frames = {key: val for key, val in true_frames.items()
                                   if (key >= start_frame) and (key <= frame)}

                # Добавляем новую подзадачу в итоговый список:
                subtasks.append((sub_df, file_path, sub_true_frames))

                start_frame = None  # Закрываем последовательность

            # Если при этом последовательность даже не открывалась, то выводим
            # ошибку:
            else:
                debug_str += ' <- Конец последовательности в метках без'
                debug_str += ' начала в кадре №%d(%d)' % (frame, true_frame)
                debug_str += ' проекта "%s"' % proj_name
                debug_str += ' задачи "%s"!\n' % task_name
                if debug_mode:
                    print(debug_str)
                    break
                else:
                    raise ValueError(debug_str)

    # После конца цикла проверяем, закрыта ли последняя последовательность:
    else:
        if start_frame is not None:
            debug_str += ' <- Ненайден конец последовательности'
            debug_str += ' проекта "%s"' % proj_name
            debug_str += ' задачи "%s"!\n' % task_name
            if debug_mode:
                print(debug_str)
            else:
                raise ValueError(debug_str)

    return subtasks


def split_task_by_labels(task, debug_mode: 'Режим отладки' = True):
    '''
    Дробит все подзадачи текущей задачи, согласно меткам
    "scene-start" и "scene-finish" в датафрейме.
    Применять надо строго после интерполяции контуров.
    '''
    splitted_task = []
    for subtask in task:
        splitted_task += split_subtask_by_labels(subtask, debug_mode)

    return splitted_task


def split_true_frames_in_tasks(tasks                                                                                    ,
                               debug_mode: 'Режим отладки выводит отладочную информацию, не прерывая выполнение' = True ,
                               desc      : 'Название статусбара'                                                 = None ):
    '''
    Распиливает df на части, согласно меткам 'scene-start', 'scene-finish'.
    '''
    # Параллельная обработка данных:
    debug_modes = [debug_mode] * len(tasks)
    tasks = mpmap(split_task_by_labels, tasks, debug_modes, desc=desc)

    # Выбрасываем пустые задачи:
    tasks = [task for task in tasks if len(task)]

    return tasks


def fuse_multipoly_in_df(points):
    '''
    Грубо избавляет контур от повторяющихся точек, указывающих на
    # многоконтурность.
    '''
    return CVATPoints(points).fuse_multipoly().flatten()


def smart_fuse_multipoly_in_df(points):
    '''
    Избавляет контур от повторяющихся точек, указывающих на многоконтурность.
    '''
    return CVATPoints(points).smart_fuse_multipoly().flatten()


def drop_unlabeled_frames(tasks):
    '''
    Отбрасывает в задачах кадры, не содежращие в разметке ни одного объекта.
    Полезно для отбрасывания неразмеченных сюжетов, если в кадрах не может не
    быть объектов.
    '''
    filtred_tasks = []  # Инициируем список отфильтрованных задач

    for task in tasks:  # Перебираем задачи из списка
        filtred_task = []  # Инициируем отфильтрованную задачу

        # Перебираем список подзадач:
        for df, file, true_frames in task:
            labeled_true_frames = df['true_frame'].unique()
            new_true_frames = {k: v for k, v in true_frames.items()
                               if k in labeled_true_frames}

            filtred_task.append((df, file, new_true_frames))

        filtred_tasks.append(filtred_task)

    return filtred_tasks


def get_total_frames(tasks):
    '''
    Подсчёт общего числа кадров в списке задач.
    '''
    total_frames = 0
    for raw_task in tasks:
        for subtask in raw_task:
            total_frames += len(subtask[2])

    return total_frames


def task_fuse_multipoly(task, smart=True):
    '''
    Постобработка задачи.
    Используется для избавления контуров от повторяющихся точек если
    многоконтурность исключена.
    '''
    # Инициируем новый список подзадач:
    task_ = []

    # Определяем функцию преобразования:
    func = smart_fuse_multipoly_in_df if smart else fuse_multipoly_in_df

    # Перебираем все подзадачи:
    for df, file, true_frames in task:

        if df is not None:

            # Дублируем текущий датафрейм чтобы не редактировать оригинал:
            df = df.copy()

            # Формируем маску для выделения только тех объектов, чьи точки
            # реально являются контурами:
            mask = df['type'] == 'polygon'

            # Применяем преобразование для всех контуров датафрейма:
            df.loc[mask, 'points'] = df.loc[mask, 'points'].apply(func)

        # Вносим обновлённую подзадачу в конечный список:
        task_.append([df, file, true_frames])

    return task_


def tasks_fuse_multipoly(tasks: 'Задачи'                    ,
                         smart: 'Аккуратный режим'    = True,
                         desc : 'Название статусбара' = None):
    '''
    Постобработка всех задач.
    Используется для избавления контуров от повторяющихся точек если
    многоконтурность исключена.
    '''
    return mpmap(task_fuse_multipoly, tasks, [smart] * len(tasks), desc=desc)


def cvat_backups2tasks(unzipped_cvat_backups_dir):
    '''
    Формирует список из распарсенных и обработанных задач из папки с
    распакованными версиями CVAT-бекапов. Постаброботка включает в себя
    интерполяцию контуров и разбиение на сцены по меткам.
    '''
    # Парсим распакованные датасеты и формируем список данных для каждого
    # видео:
    raw_tasks = cvat_backups2raw_tasks(
        unzipped_cvat_backups_dir, desc='Парсинг распакованных CVAT-датасетов')

    # Удаляем повторы в разметке:
    print('')  # Отступ
    raw_tasks = drop_label_duplicates_in_tasks(
        raw_tasks, desc='Удаление повторов в разметке')

    # Интерполируем сегменты во всех неключевых кадрах:
    print('')  # Отступ
    interp_tasks = interpolate_tasks_df(
        raw_tasks, desc='Интерполяция контуров в неключевых кадрах')

    # Разрезаем последовательности кадров, опираясь на метки 'scene-start',
    # 'scene-finish'. Этот шаг можно пропустить если датасет не содержит меток
    # 'scene-start' и 'scene-finish':
    print('')  # Отступ
    splited_tasks = split_true_frames_in_tasks(
        interp_tasks, desc='Расщепление смонтированных последовательностей')

    # Постобработка задач:
    print('')  # Отступ
    tasks = tasks_fuse_multipoly(
        splited_tasks, smart=False,
        desc='Очистка контуров от повторяющихся точек')

    # Сортировка задач для воспроизводимости результата:
    tasks = sort_tasks(tasks)

    return tasks


def yandex2subtask(json_file, file):
    '''
    Парсит разметку Яндекс Крауда.
    '''
    # Читаем файл с разметкой:
    with open(json_file, 'r', encoding='utf-8') as f:
        ya_task = json.load(f)

    # Упорядочиваем разметку по возрастанию номеров кадра:
    frames = []
    for frame_data in ya_task:
        inputValues = frame_data.get('inputValues', frame_data)
        if inputValues is None:
            frames.append(frame_data['frame'])
        else:
            frames.append(inputValues['frame'])

    ya_task = [ya_task[sorted_ind] for sorted_ind in np.argsort(frames)]

    # Инициируем переменные, которыми будем описывать задачу:
    dfs = []
    true_frames = {}

    # Читаем каждый кадр:
    img_buffer = ImReadBuffer()
    for frame, frame_data in enumerate(ya_task):

        # Читаем входные и (если есть) выходные данные:
        inputValues = frame_data.get('inputValues', frame_data)
        outputValues = frame_data.get('outputValues', inputValues)
        if inputValues is None:
            inputValues = outputValues = frame_data

        # Извлекаем основнух инфу из входных и выходных данных:
        true_frame = inputValues['frame']
        polygons = outputValues['polygons']

        # Получаем размер очередного кадра:
        img = img_buffer(file, true_frame)
        if img is None:
            print(true_frame)
            raise
        imsize = img.shape[:2]

        # Перебираем каждый сегмент в кадре:
        for polygon in polygons:

            # Читаем имя класса и геометрию:
            label = polygon['label']
            points = polygon['points']

            # Код пока не умеет работать ни с чем, кроме многоугольников:
            assert polygon['shape'] == 'polygon'

            # Создаём и добавляем очередную строку датафрейма в список:
            points = [(point['left'], point['top']) for point in points]
            segment = CVATPoints.from_yoloseg(points, imsize=imsize)
            segment = segment.fuse_multipoly()  # Убираем повторы в контуре
            dfs.append(segment.to_dfrow(label=label,
                                        frame=frame,
                                        true_frame=true_frame,
                                        source='Yandex'))

        # Дополняем словарь номеров используемых кадров:
        true_frames[frame] = true_frame

    # Объединяем строки в единый датафрейм:
    df = pd.concat(dfs)

    return df, file, true_frames


# Вовзращает значение заданного столбца.
# Во всех строках это значение должно быть одинаковым:
def get_single_val_in_df(df, column):

    # Получаем все уникальные значения столбца:
    values = df[column].unique()

    # Все значения должны быть одинаковыми:
    assert len(values) == 1

    # Берём это единственное значение.
    return values[0]


def subtask2xml(subtask, xml_file=None):
    '''
    Сохраняет cvat-подзадачу в XML-файл с аннотациями, принимаемыми интерфейсом
    CVAT в качестве разметки.
    '''
    # Расщепление подзадачи на составляющие:
    df, file, true_frames = subtask

    # Датафрейм разметки на:
    df_tracks = df[df['track_id'].notna()]  # Треки
    df_shapes = df[df['track_id']. isna()]  # Формы

    # Инициализируем XML-структуру:
    annotations = ET.Element('annotations')

    # Сохраняем треки (tracks):
    for track_id in df_tracks['track_id'].unique():

        # Весь датафрейм для текущего трека:
        df_track = df_tracks[df_tracks['track_id'] == track_id]

        # Получаем метку и группу объекта:
        label = get_single_val_in_df(df_track, 'label')
        group = str(int(get_single_val_in_df(df_track, 'group')))
        source = get_single_val_in_df(df_track, 'source')

        # Инициируем аннотацию изображения:
        track = ET.SubElement(annotations, 'track', id=str(track_id),
                              label=label, source=source, group_id=group)

        # Перебор всех вхождений объекта текущего трека в
        # видеопоследотвательность:
        for dfrow in df_track.iloc:

            # Получаем XML-параметры трека в текущем кадре:
            args, kwargs = CVATPoints.from_dfrow(dfrow).xmlparams()

            # Вносим описание трека в текущем кадре в XML-структуру:
            ET.SubElement(track                                 ,
                          frame    = str(    dfrow['frame'   ] ),
                          outside  = str(int(dfrow['outside' ])),
                          occluded = str(int(dfrow['occluded'])),
                          keyframe = "1"                        ,
                          z_order  = str(int(dfrow['z_order' ])),
                          *args, **kwargs)

    # Сохраняем формы (shapes):
    for frame, true_frame in true_frames.items():

        # Определяем имя кадра:
        name = os.path.basename(file if isinstance(file, str)
                                else file[frame])

        # Инициируем аннотацию изображения:
        image = ET.SubElement(annotations, 'image', id=str(frame), name=name)

        # Проходим по всем строкам датафрейма:
        for dfrow in df_shapes[df_shapes['frame'] == frame].iloc:

            # Получаем XML-параметры контуров:
            args, kwargs = CVATPoints.from_dfrow(dfrow).xmlparams()

            # Вносим описание контура в XML-структуру:
            ET.SubElement(image                                 ,
                          label    =         dfrow['label'   ]  ,
                          occluded = str(int(dfrow['occluded'])),
                          source   =         dfrow['source'  ]  ,
                          z_order  = str(int(dfrow['z_order' ])),
                          group_id = str(int(dfrow['group'   ])),
                          *args, **kwargs)

    # Возвращаем XML-структуру, если файл для записи не указан:
    if xml_file is None:
        return annotations

    # Пишем XML-структуру в файл:
    ET.ElementTree(annotations).write(xml_file,
                                      encoding="utf-8",
                                      xml_declaration=True)

    return xml_file


def file2unlabeled_subtask(file, check_frames=False):
    '''
    Создаёт задачу без разметки, опираясь на имя файла/файлов.
    '''
    # Если уже передан список, ничего не делаем:
    if isinstance(file, (list, tuple)):
        pass

    # Если передан путь к дирректории, то берём все файлы из неё:
    elif os.path.isdir(file):
        file = [os.path.join(file, _) for _ in sorted(os.listdir(file))]

    # Если передан путь к файлу, до делаем из него список:
    elif os.path.isfile(file):
        file = [file]

    else:
        raise FileNotFoundError(f'Файл "{file}" не найден!')

    # Оставляем в списке только существующие файлы:
    file = list(filter(os.path.isfile, file))

    # Оставляем только подходящие по типу файлы:
    cvat_data_exts = cv2_vid_exts | cv2_img_exts
    file = [_ for _ in file
            if os.path.splitext(_)[-1].lower() in cvat_data_exts]

    # Берём первый элемент списка, если он единственный:
    if len(file) == 1:
        file = file[0]

    # Генератор, облегчающий процесс подсчёта числа кадров:
    vg = VideoGenerator(file)

    # Буквалльно читаем все данные, чтобы точно знать общее число кадров:
    if check_frames:
        total_frames = 0
        for frame in vg:
            if frame is None:
                break
            total_frames += 1

    # Оцениваем совокупное число кадров по косвенным данным:
    else:
        total_frames = len(vg)

    # Составляем словарь кадров без прореживания:
    true_frames = {_: _ for _ in range(total_frames)}

    return None, file, true_frames


def dir2unlabeled_tasks(path, check_frames=False):
    '''
    Перебирае все фото и видео из заданной папки и её подпапок,
    создавая из них список неразмеченных задач.
    Используется для авторазметки неупорядоченных данных.

    Каждый видеофайл размещается в отдельной задаче.
    Все изображения, находящиеся в одной дирректории размещаются в задаче,
    соответствующей этой дирректории.
    '''

    # Инициируем список задач файлами из корневой дирректории:
    tasks = []
    img_files = []  # Список изображений текущей дирректории
    for file in sorted(os.listdir(path)):

        ext = os.path.splitext(file)[1].lower()  # Расширение файла
        file = os.path.join(path, file)          # Полный путь до файла

        # Каждое найденное видео сразу заносится в отдельную задачу:
        if ext in cv2_vid_exts:
            subtask = file2unlabeled_subtask(file, check_frames)
            tasks.append([subtask])

        # Каждое найденное изображение заносится в список:
        elif ext in cv2_img_exts:
            img_files.append(file)

        # Если это папка - делаем рекурсию:
        elif os.path.isdir(file):
            tasks += dir2unlabeled_tasks(file, check_frames)

    # Все найденные изобржаения текущей папки собираем в одну подзадачу:
    if img_files:
        subtask = file2unlabeled_subtask(img_files, check_frames)
        tasks.append([subtask])

    return tasks


def task_auto_annottation(task,
                          img2df,
                          label=None,
                          store_prev_annotation=True,
                          desc=None):
    '''
    Применяет img2df для автоматической разметки бекапа
    cvat-задачи и сохраняет результат в файл annotation_file.
    '''
    # Инициализируем конечный список подзадач:
    task_ = []

    # Инициализируем буфер для чтения изображений:
    img_buffer = ImReadBuffer()

    # Перебор подзадач:
    for subtask_ind, (df, file, true_frames) in enumerate(task, 1):

        # Инициализируем список датафреймов для последующего объединения:
        frame_dfs = []

        # Уточняем название статусбара:
        desc_ = f'{desc} ({subtask_ind} / {len(task)})' \
            if len(task) > 1 else desc

        # Перебор кадров:
        for frame, true_frame in tqdm(true_frames.items(),
                                      desc=desc_,
                                      disable=desc is None):

            # Читаем очередной кадр и переводим его в RGB:
            img = img_buffer(file, true_frame)[..., ::-1]

            # Получаем датафрейм, содержащий результат авторазметки:
            frame_df = img2df(img)

            # Переходим к следующему кадру, если в этом меток нет:
            if frame_df is None:
                continue

            # Коррекция значений в столбце метки класса:
            if label:
                frame_df.loc[frame_df['label'].isna(), 'label'] = label
            # Коррекция значений в столбцах номеров кадров:
            frame_df.loc[:, 'frame'] = frame            # номер прореженного кадра
            frame_df.loc[:, 'true_frame'] = true_frame  # номер кадра

            # Добавление очередного датафрейма в общий список:
            frame_dfs.append(frame_df)

        # Формируем объединённый датафремй, содержащий или исключающий исходную
        # разметку:
        df = concat_dfs([df] + frame_dfs
                        if store_prev_annotation
                        else frame_dfs)

        # Внесение очередной подзадачи в итоговую задачу:
        task_.append((df, file, true_frames))

    return task_


def tasks_auto_annottation(tasks,
                           img2df,
                           label=None,
                           store_prev_annotation=True,
                           num_procs=1,
                           desc='Авторазметка',
                           **kwargs):
    '''
    Применеие авторазметки ко всем задачам.
    '''
    # Включаем num_procs и desc в остальные параметры kwargs для mpmap:
    kwargs = kwargs | {'num_procs': num_procs, 'desc': desc}

    # Выполняем авторазметку:
    return mpmap(task_auto_annottation,
                 tasks,
                 [img2df] * len(tasks),
                 [label] * len(tasks),
                 [store_prev_annotation] * len(tasks),
                 **kwargs)


def cvat_backup_task_dir2auto_annotation_xml(cvat_backup_task_dir,
                                             img2df,
                                             label=None,
                                             store_prev_annotation=True,
                                             xml_file=None,
                                             desc=None):
    '''
    Выполняет автоматическую разметку задачи по её бекапу и сохраняет в
    cvat-совместимый xml-файл.
    '''
    # Размещаем разметку в папке с задачей, если файл для сохранения явно не
    # указан:
    if xml_file is None:
        xml_file = os.path.join(cvat_backup_task_dir, 'annotation.xml')

    # Читаем имеющуюся разметку:
    task = cvat_backup_task_dir2task(cvat_backup_task_dir)

    # Пока работает только для задач, состоящих лишь из одной подзадачи:
    assert len(task) == 1

    # Выполняем доразметку:
    task_ = task_auto_annottation(task, img2df, label,
                                  store_prev_annotation=store_prev_annotation,
                                  desc=desc)

    # AФормируем один или несколько xml-файлов для каждой подзадачи:

    # Если подзадача всего одна, то сохраняем её в файле с заданным именем:
    if len(task_) == 1:
        subtask2xml(task_[0], xml_file)

    # Если подзадач несколько:
    else:

        # Перебор подзадач:
        for job_ind, subtask in enumerate(task, 1):

            # Разбиваем путь до xml-файла на имя и разширение:
            xml_file_name, xml_file_ext = os.path.splitext(xml_file)

            # Сохраняем разметку в файл с суффиксом "_{номер подзадачи}":
            subtask2xml(task_[0], xml_file_name + f'_{job_ind}')

    return task_


def cvat_backup_dir2auto_annotation_xmls(cvat_backup_dir,
                                         img2df,
                                         label='unlabeled',
                                         store_prev_annotation=True,
                                         xml_dir=None,
                                         num_procs=1,
                                         desc=None):
    '''
    Выполняет автоматическую разметку распакованного бекапа cvat-датасета и
    сохраняет в cvat-совместимые xml-файлы.
    '''
    # Инициируем аргументы для mpmap:
    cvat_backup_task_dirs  = []
    img2dfs                = []
    labels                 = []
    store_prev_annotations = []
    xml_files              = []

    # Составляем аргументы для mpmap:

    # Перебираем все вложенные в папку с датасетом директории:
    for cvat_backup_task_dir in os.listdir(cvat_backup_dir):
        # print(f'Обробатываем задачу из папки "{cvat_backup_task_dir}" ...')

        # Уточняем полный путь до поддиректории:
        cvat_backup_task_dir = os.path.join(cvat_backup_dir,
                                            cvat_backup_task_dir)

        # Если это не папка, то пропускаем:
        if not os.path.isdir(cvat_backup_task_dir):
            continue

        # Определяем полный путь до xml-файла:
        xml_file = cvat_backup_task_dir + '.xml' if xml_dir is None else \
            os.path.join(xml_dir,
                         os.path.basename(cvat_backup_task_dir) + '.xml')
        # Если путь до папки с итоговой разметкой задан, то каждый xml-файл
        # размещается в ней под именем папки с соответствующей задачей. Если
        # путь не задан, то каждый xml-файл размещается в папке с
        # соответствующей задачей.

        # Создаём папку для xml, если надо:
        xml_dir = os.path.split(xml_file)[0]
        if os.path.isdir(xml_dir):
            mkdirs(xml_dir)

        # Вносим каждый из параметров в свой список:
        cvat_backup_task_dirs .append(cvat_backup_task_dir)
        img2dfs               .append(img2df)
        labels                .append(label)
        store_prev_annotations.append(store_prev_annotation)
        xml_files             .append(xml_file)

    # Выполняем авторазметку в параллельном режиме:
    return mpmap(cvat_backup_task_dir2auto_annotation_xml,
                 cvat_backup_task_dirs,
                 img2dfs,
                 labels,
                 store_prev_annotations,
                 xml_files,
                 xml_files if desc else [None] * len(xml_files),
                 num_procs=num_procs,
                 desc=desc)


def df2masks(df, imsize, saving_memory=False):
    '''
    Переводит датафрейм с объектами в список масок.
    '''
    # Определяем индекс столбца с контурами:
    points_col_ind = np.argwhere(df.columns == 'points').flatten()
    assert len(points_col_ind) == 1
    points_col_ind = points_col_ind[0]

    # Создаём и наполняем список масок:
    masks = []
    for dfrow in df.iloc:
        points = CVATPoints.from_dfrow(dfrow, imsize=imsize)
        mask = DelayedInit(Mask, kwargs_func=points.to_Mask_kwargs) \
            if saving_memory else Mask(**points.to_Mask_kwargs())
        masks.append(mask)

    return masks


def concat_dfs(*args):
    '''
    Объединяет датафреймы/строки и их списки в один общий датафрейм.
    '''
    # Обрабатываем список аргументов (с рекурсией, если надо):
    dfs = []
    for df in args:
        if isinstance(df, pd.core.frame.DataFrame):
            dfs.append(df)
        elif isinstance(df, pd.core.series.Series):
            dfs.append(pd.DataFrame(df).T)
        elif isinstance(df, (tuple, list, set)):
            dfs.append(concat_dfs(*df))
        elif df is None:
            pass
        else:
            raise ValueError(f'Передан неожиданный тип данных: {type(df)}')

    # Возвращаем результат:
    if len(dfs) == 0:
        return None
    elif len(dfs) == 1:
        return dfs[0]
    else:
        return pd.concat(dfs, ignore_index=True)


def hide_skipped_objects_in_df(df, true_frames):
    '''
    Корректно скрывает объекты, в тех кадрах, где их сегменты не прописаны
    явным образом. Т.е. там, где раньше CVAT бы интерполировал два состояния
    между ключевыми кадрами, теперь объект будет скрыт.

    Полезно в случае, если датафрейм был как-то сгенерирован из данных, не
    исопльзующих интерполяцию. Например, при авторазметке, где отстутствие
    метки объекта в конкретном кадре означает что объект не был обнаружен.
    '''
    # Определяем номер последнего кадра:
    last_frame = max(true_frames.keys())

    # Инициируем список строк с замыкающими объектами:
    hidden_dfs = []

    # Перебираем каждый трек:
    track_ids = df['track_id'].unique() if len(df) else []
    for track_id in track_ids:

        # Формы пропускаем:
        if track_id is None:
            continue

        # Формируем датафрейм текущего трека:
        track_df = df[df['track_id'] == track_id]

        # Перебираем все объекты трека:
        for ind in range(len(track_df)):
            df_row = track_df.iloc[ind, :].copy()

            # Объект должен быть виден:
            assert ~df_row['outside']

            # Определяем номер кадра:
            frame = df_row['frame']

            # Последние кадры последовательности не нуждаются в замыкании:
            if frame == last_frame:
                continue

            # Если запись следующего кадра существует, то замыкание не
            # требуется:
            if (track_df['frame'] == frame + 1).any():
                continue

            # Создаём замыкание и заносим его в список:
            df_row['outside'] = True
            df_row['frame'] = frame + 1
            df_row['true_frame'] = true_frames[frame + 1]
            hidden_dfs.append(df_row)

    # Добавляем замыкающие объекты к основному датафрейму и сортируем по
    # номеру кадра:
    df = concat_dfs([df] + hidden_dfs)
    return df.sort_values('frame')


def split_df_by_visibility(df):
    '''
    Разделяет датафрейм объектов на 2 по признаку видимости.
    '''
    # Инициируем итоговые списки объектов:
    visible_df = []
    invisible_df = []

    # Расфасовываем каждый объект исходного списка по итоговым:
    for dfrow in df.iloc:
        if dfrow['outside']:
            invisible_df.append(dfrow)
        else:
            visible_df.append(dfrow)

    # Собираем датафреймы из списков:
    visible_df = concat_dfs(visible_df)
    invisible_df = concat_dfs(invisible_df)

    if visible_df is None:
        visible_df = new_df()
    if invisible_df is None:
        invisible_df = new_df()

    return visible_df, invisible_df


def apply_mask_processing2df(df, imsize, processing, mpmap_kwargs={}):
    '''
    Растеризирует каждую маску датафрейма и выполняет покадровую фильтрацию.
    Обновлённые маски векторизируются и заносятся обратно в датафрейм.
    Требуется, чтобы набор фильтров не менял число и порядок масок!
    '''
    # Получаем столбец с номерами кадров:
    df_frames = df['frame']

    # Получаем список всех используемых в датафрейме кадров:
    unique_frames = sorted(df_frames.unique())

    # Если передан датафрейм одного кадра, то обрабатываем его целиком:
    if len(unique_frames) == 1:
        frame = unique_frames[0]
        masks = df2masks(df, imsize)

        # Применяем обработку или цепочку обработок:
        if isinstance(processing, (list, tuple)):
            for proc in processing:
                masks = proc(masks)
        else:
            masks = processing(masks)

        if len(masks) != len(df):
            raise ValueError(
                f'Фильтрация изменила число масок на {frame}-м кадре!')

        # Определяем номер столбца с точками:
        points_ind = get_column_ind(df, 'points')

        # Векторизируем обработанные контуры и вносим их обратно в датафрейм:
        for ind, mask in enumerate(masks):
            points = CVATPoints.from_mask(mask.array)
            if points is not None:
                points = points.flatten()
            df.iat[ind, points_ind] = points

        return df

    # Если в датафрейме несколько кадров, то обрабатываем их параллельно:
    elif len(unique_frames) > 1:

        # Разбиваем датафрейм на кадры:
        dfs = [df[df_frames == frame] for frame in unique_frames]

        # Формируем список списков масок по каждому кадру:
        dfs = mpmap(apply_mask_processing2df,
                    dfs,
                    [imsize] * len(dfs),
                    [processing] * len(dfs),
                    **mpmap_kwargs)

        return pd.concat(dfs)

    # Если в датафрейме нет ни одного объекта, возвращаем его без изменений:
    else:
        return df


def df2img(df, label2color, imsize, img=None):
    '''
    Отрисовка всех сегментов на одном изображении.
    '''
    if img is None:
        img = np.zeros(list(imsize)[:2] + [3], dtype=np.uint8)

    for dfrow in df.iloc:
        points = CVATPoints.from_dfrow(dfrow)
        color = label2color[dfrow['label']]
        img = points.draw(img, color=color, thickness=-1)

    return img


# ВременнАя фильтрация:
def subtask_shapes2tracks(subtask,
                          minIoU=0.6,
                          depth=20,
                          untracked_label='unlabeled',
                          cut_tails=False,
                          drop_untracked_shapes=True,
                          desc='Трекинг несвязных форм в подзадаче',
                          num_procs=0,
                          memory_saving=True):
    '''
    Трекинг несвязных форм в подзадаче.
    Полезен при доразметке видео после прогона через автоматическую
    разметку, которая работает покадрово (генерирует только формы).
    Для лучшего результата следует корректно разметить первый кадр.
    Остальные кадры будут размечены по аналогии.
    '''
    # Расщепление подзадачи на составляющие:
    df, file, true_frames = subtask

    # Номера столбцов датафрейма:
    track_id_ind   = get_column_ind('track_id')    # track_id
    label_ind      = get_column_ind('label')       # label
    frame_ind      = get_column_ind('frame')       # frame
    true_frame_ind = get_column_ind('true_frame')  # true_frame
    # Нужны для доступа к ячейкам через df.iloc.

    # Число кадров в последовательности:
    seq_len = len(true_frames)

    # Упорядоченный список всех непрореженных кадров:
    frames_list = sorted(list(true_frames.keys()))

    # Разбиваем общий датафрейм на множество покадровых,
    # содежащих только формы (без треков):
    shape_dfs = [df[df['track_id'].isna() & (df['frame'] == frame)]
                 for frame in frames_list]
    # Между объектами из этих датафреймов и будет устанавливаться связь.

    # Определяем размеры каждого кадра:
    with ImReadBuffer() as buffer:
        imsizes = [buffer(file, frame).shape[:2] for frame in frames_list]

    # Получаем списки масок по каждому из кадров:
    if memory_saving:  # Список масок с отложенной инициацией:
        masks = mpmap(df2masks, shape_dfs, imsizes, [True] * len(imsizes),
                      num_procs=1)
        # Полезно для экономии памяти.
    else:  # Список масок с полной загрузкой памяти:
        masks = mpmap(df2masks, shape_dfs, imsizes, num_procs=num_procs)

        # Сразу индексируем площади и обрамляющие прямоугольники для каждой
        # из масок:
        for frame_masks in masks:
            for mask in frame_masks:
                mask.rectangle()
                mask.area()
        # Это позволит быстрее исключать пересечения сегментов при оценке
        # связностей.

    # Датафрейм с уже существующими треками.
    track_df = df[df['track_id'].notna()]
    # С ним будет объединён результат выстраивания цепочек форм в треки.

    # Перебираем все возможные пары номеров кадров,
    # отстаящих друг от друга не более чем на depth:
    keys            = []
    cur_masks_list  = []
    next_masks_list = []
    desc_list       = []
    num_procs_list  = []

    # Формируем список задач для вычисления матриц связностей через mpmap:
    for cur_ind in range(seq_len - 1):
        for next_ind in range(cur_ind + 1, min(cur_ind + depth + 1, seq_len)):
            keys.append((cur_ind, next_ind))         # Ключи для последующего преобразования списка в словарь
            cur_masks_list .append(masks[ cur_ind])  # Списки масок текущего кадра
            next_masks_list.append(masks[next_ind])  # Списки масок одного из последующих кадров
            desc_list      .append(None)             # Отключаем статусбар дочерних процессов
            num_procs_list .append(1)                # Отключаем параллельность в дочерних процессах

    # Оценка вычислительной сложности каждой из задач для mpmap:
    complexity_list = [len(masks1) * len(masks2)
                       for masks1, masks2 in
                       zip(cur_masks_list, next_masks_list)]

    # Аргументы mpmap сортируются в порядке убывания сложности задачи:
    keys, *mpmap_args = reorder_lists(np.argsort(complexity_list)[::-1],
                                      keys,
                                      cur_masks_list,
                                      next_masks_list,
                                      desc_list,
                                      num_procs_list)
    # Полезно для сокращения общего времени параллельных вычислений в mpmap.

    # Параллельное вычисление матриц связностей:
    IoUmats = mpmap(build_masks_IoU_matrix, *mpmap_args, desc=desc,
                    num_procs=num_procs)

    # Преобразуем результаты параллельных вычислений из списка в словарь:
    IoUmats = {key: val for key, val in zip(keys, IoUmats)}

    # Начинаем строить цепочки наследования:

    # Определяем номер, с которого начинаются ещё не занятые track_id:
    track_id = df['track_id'].max()
    if np.isnan(track_id):
        track_id = 0  # Устанавливаем в 0, если они не использовались вообще

    # Получаем датафрейм первого кадра:
    first_df = shape_dfs[0]

    # Выделяем те формы в первом кадре, для которых установлен класс:
    untracked_shapes_mask = first_df.iloc[:, label_ind] != untracked_label

    # Преобразуем все размеченные формы первого кадра в треки
    # (расставляем номера для их track_id):
    first_df.iloc[untracked_shapes_mask,
                  track_id_ind] = range(track_id,
                                        track_id + untracked_shapes_mask.sum())

    # Инициируем списки флагов подтверждения для каждой из масок:
    isexists = [np.ones(len(_), dtype=bool) for _ in masks]

    # Перебираем все рассмотренные в IoUmats связи:

    # Перебираем номера текущих кадров:
    for cur_ind in range(1, seq_len):

        # Инициируем список матриц связностей, подлежащих конкатенации:
        cur_IoUmats = []

        # Перебираем номера предыдущих кадров:
        for prev_ind in range(max(0, cur_ind - depth), cur_ind):

            # Получаем очередную матрицу связности сегментов текущего ...
            # ... кадра с сегментами одного из предыдущих кадров:
            IoUmat = IoUmats[(prev_ind, cur_ind)]

            # Обнуляем все IoU, оказавшиеся ниже порогового значения:
            IoUmat[IoUmat < minIoU] = 0

            # Обнуляем строки всех неподтверждённых объеков:
            IoUmat[np.invert(isexists[prev_ind]), :] = 0

            # Добавляем таблицу связностей в список матриц для конкатинации:
            cur_IoUmats.append(IoUmat)

        # Конкатинация матриц:
        cur_IoUmats = np.vstack(cur_IoUmats)

        # Обнуляем столбцы всех неподтверждённых объеков:
        cur_IoUmats[:, np.invert(isexists[cur_ind])] = 0

        # Применяем венгерский алгоритм, выполняющий оптимальные назначения:
        prev_mask_inds, cur_mask_inds = linear_sum_assignment(cur_IoUmats,
                                                              maximize=True)
        # Сегментам из текущего кадра ставятся в соответсвие ...
        # ... сегменты из предыдущих кадров (см. двудольный граф).

        # Сортируем индексы связных сегментов в порядке убывания связей:
        sorted_inds = np.argsort(
            cur_IoUmats[prev_mask_inds, cur_mask_inds])[::-1]
        prev_mask_inds, cur_mask_inds = reorder_lists(sorted_inds,
                                                      prev_mask_inds,
                                                      cur_mask_inds)

        # Инициируем множество уже использованных треков:
        track_ids2drop = set()

        # Перебор индексов всех найденных пар сегментов:
        for prev_mask_ind, cur_mask_ind in zip(prev_mask_inds, cur_mask_inds):

            cur_IoU = cur_IoUmats[prev_mask_ind, cur_mask_ind]
            # Если связность сегментов существенная:
            if cur_IoU:

                # Определяем номер предыдущего кадра и номер объекта в нём:

                # Перебираем номера предыдущих кадров:
                for prev_ind in range(max(0, cur_ind - depth), cur_ind):
                    shifted_prev_mask_ind = prev_mask_ind - \
                                            len(isexists[prev_ind])
                    if shifted_prev_mask_ind < 0:
                        break
                    prev_mask_ind = shifted_prev_mask_ind
                # Это как бы реиндексация, необходимая для получения координат
                # ячейки таблицы связностей, ещё до конкатенации.

                # Убеждаемся,что реиндексацию провели верно:
                assert cur_IoU == IoUmats[(prev_ind, cur_ind)][prev_mask_ind,
                                                               cur_mask_ind]

                # 
                track_id = shape_dfs[prev_ind].iloc[prev_mask_ind,
                                                    track_id_ind]

                if track_id in track_ids2drop:
                    isexists[cur_ind][cur_mask_ind] = False

                else:
                    track_ids2drop.add(track_id)

                    # Увязываем сегмент текущего кадра в трек сегмента из
                    # предыдущих кадров:
                    for colimn_ind in {label_ind, track_id_ind}:
                        shape_dfs[cur_ind].iloc[cur_mask_ind, colimn_ind] = \
                            shape_dfs[prev_ind].iloc[prev_mask_ind, colimn_ind]

            # Если связность незначительная, то помечаем ...
            # ... сегмент текущего кадра, как неподтверждённый:
            else:
                isexists[cur_ind][cur_mask_ind] = False

    # "Отрубаем хвосты" трекам, если надо:
    if cut_tails:

        # Инициируем множество уже рассмотренных треков:
        excluded_track_ids = set(shape_dfs[-1].iloc[:, track_id_ind])
        # Добавляем в него все "дожившие" до последнего кадра треки.
        # Множество используется для отличия уже рассмотренных треков ...
        # ... от "новых", которые как раз и надо будет "обрубать".

        next_frame      = shape_dfs[-1].iloc[0,      frame_ind]
        next_true_frame = shape_dfs[-1].iloc[0, true_frame_ind]
        next_ind = seq_len - 1
        # Перебираем все кадры в обратном порядке, кроме крайних:
        for cur_ind in reversed(range(1, seq_len - 1)):

            # 
            additional_dfs = []

            # Перебираем все объекты в данном кадре:
            for row in shape_dfs[cur_ind].iloc:

                # Получаем track_id текущего объекта:
                track_id = row.iloc[track_id_ind]

                # Пропускаем не треки:
                if track_id is None: continue

                # Пропускаем уже рассмотренные треки:
                if track_id in excluded_track_ids: continue

                # Вносим track_id текущего объекта в множество уже
                # рассмотренных:
                excluded_track_ids.add(track_id)

                additional_df = pd.DataFrame(row).T
                additional_df[     'frame'] =      next_frame
                additional_df['true_frame'] = next_true_frame
                additional_df['outside'   ] = "1"
                additional_df['source'    ] = additional_df['source'] + ' + tacker'

                additional_dfs.append(additional_df)

            # Добавляем все "заглушки" в следующий кадр:
            shape_dfs[next_ind] = pd.concat([shape_dfs[next_ind]] + additional_dfs)

            next_frame      = row.iloc[     frame_ind]
            next_true_frame = row.iloc[true_frame_ind]
            next_ind = cur_ind

    # Объединение оставшихся сегментов и построенных треков с уже имевшимися
    # треками:
    df = pd.concat([track_df] + shape_dfs)

    # Выбрасываем оставшиеся сегменты (не включённые в трек) из результата,
    # если нужно:
    if drop_untracked_shapes:
        df = df[df['track_id'].notna()]

    # Возвращаем подзадачу с обновлённым датафреймом:
    return df, file, true_frames


def bidirectional_subtask_shapes2tracks(subtask,
                                        minIoU=0.6,
                                        depth=20,
                                        untracked_label='unlabeled',
                                        cut_tails=False,
                                        drop_untracked_shapes=True,
                                        desc='Трекинг несвязных форм',
                                        num_procs=0):
    '''
    Аналогичен subtask_shapes2tracks, но работает и с ключевым кадром
    в середине последовательности. Т.е. поиск связных сегментов ведётся
    в обоих направлениях по оси времени.
    '''
    # Расщепление подзадачи на составляющие:
    df, file, true_frames = subtask

    # Составляем список номеров ключевых кадров:
    df_frames = df['frame']
    unique_frames = sorted(df_frames.unique())
    key_frames = []
    for frame in unique_frames:
        labels = df.loc[df_frames == frame, 'label'].unique()
        if len(set(labels) - set((untracked_label,))):
            key_frames.append(frame)
    # Ключевыми считаются кадры, в которых есть не
    # untracked_label объекты.

    # Пока код работает только с одним ключевым кадром:
    assert len(key_frames) == 1
    key_frame = key_frames[0]

    # В ключевом кадре отбрасываем все untracked_label объекты:
    df = df[(df['frame'] != key_frame) |
            (df['label'] != untracked_label)]
    df_frames = df['frame']
    # На всякий случай обновляем переменную df_frames.

    # Если ключевой кадр является первым, то это частный случай:
    if key_frame == min(unique_frames):
        return subtask_shapes2tracks(subtask,
                                     minIoU,
                                     depth,
                                     untracked_label,
                                     cut_tails,
                                     drop_untracked_shapes,
                                     desc,
                                     num_procs)

    # Формируем датафреймы обоих направлений от ключевого кадра:
    print(key_frame)
    front_df = df[df_frames >= key_frame]
    back_df  = df[df_frames <= key_frame]

    # Получаем упорядоченные номера кадров для обоих направлений:
    front_frames_list      = sorted(front_df[     'frame'].unique())
    front_true_frames_list = sorted(front_df['true_frame'].unique())
    back_frames_list       = sorted( back_df[     'frame'].unique())
    back_true_frames_list  = sorted( back_df['true_frame'].unique())

    # Формируем словари перевода кадров в обоих направлениях:
    front_true_frames = dict(zip(front_frames_list, front_true_frames_list))
    back_true_frames  = dict(zip( back_frames_list,  back_true_frames_list))

    # Обращаем нумерацию кадров в датафрейме обратного направления:
    reverce_frames      = dict(zip(     back_frames_list, reversed(     back_frames_list)))
    reverce_true_frames = dict(zip(back_true_frames_list, reversed(back_true_frames_list)))
    back_df.loc[:,      'frame'] = back_df.loc[:,      'frame'].apply(reverce_frames     .get)
    back_df.loc[:, 'true_frame'] = back_df.loc[:, 'true_frame'].apply(reverce_true_frames.get)

    # Обработка каждого из направлений в отдельности:
    front_kwargs = {'depth': depth,
                    'untracked_label': untracked_label,
                    'cut_tails': cut_tails,
                    'drop_untracked_shapes': drop_untracked_shapes,
                    'desc': desc,
                    'num_procs': num_procs}
    back_kwargs = dict(front_kwargs)
    if desc:
        front_kwargs['desc'] = desc + ' (в  прямом  направлении)'
        back_kwargs['desc'] = desc + ' (в обратном направлении)'

    front_df = subtask_shapes2tracks((front_df, file, front_true_frames),
                                     **front_kwargs)[0]
    back_df = subtask_shapes2tracks((back_df, file, back_true_frames),
                                    **back_kwargs)[0]

    # Возвращаем обратному направлению прямую нумерацию кадров:
    reverce_frames      = dict(zip(reversed(     back_frames_list),      back_frames_list))
    reverce_true_frames = dict(zip(reversed(back_true_frames_list), back_true_frames_list))
    back_df.loc[:,      'frame'] = back_df.loc[:,      'frame'].apply(reverce_frames     .get)
    back_df.loc[:, 'true_frame'] = back_df.loc[:, 'true_frame'].apply(reverce_true_frames.get)

    # Объединяем датафреймы обоих направлений:
    df = pd.concat([back_df[back_df['frame'] != key_frame], front_df])

    return df, file, true_frames


def tasks2_train_val_test_other(tasks):
    '''
    Разделение задач на обучающую, проверочную и тестовую
    подвыборки по указанному типу для каждой задачи в самом CVAT.

    Все задачи, не отнесённые в одну из трёх основных подвыборок
    будут занесены в отдельный словарь other_tasks_dict
    (имя -> [неприкаянные_задачи]).
    '''
    # Инициализируем списки итоговых подвыборок:
    train_tasks, val_tasks, test_tasks, other_tasks_dict = [], [], [], {}
    # train_tasks_, val_tasks_, test_tasks_ = [], [], []

    # Перебор задач:
    for task in tasks:

        # Инициализация множества папок, в которых находятся задачи:
        task_dirs = set()
        # Такая папка должна быть только одна, но это надо ...
        # ... проверить, для чего и создётся данное можество.

        # Перебор подзадач:
        for df, file, true_frames in task:

            # Вносим во множество папок для задач все папки, содержащие
            # подзадачи текущей задачи:
            if isinstance(file, (tuple, list, set)):
                task_dirs |= set([os.path.dirname(os.path.dirname(_))
                                  for _ in file])

            elif isinstance(file, str):
                task_dirs.add(os.path.dirname(os.path.dirname(file)))

        # У всех файлов одной задачи должна быть одна общая папка:
        assert len(task_dirs) == 1
        task_dir = task_dirs.pop()  # Берём эту единственную папку

        # Читаем метаданные задачи:
        with open(os.path.join(task_dir, 'task.json'),
                  'r', encoding='utf-8') as f:
            task_desc = json.load(f)
        subset = task_desc['subset']  # Название подвыборки
        name = task_desc['name']      # Название задачи

        # Выясняем, к какой подвыборке принадлежит текущая задача:
        if subset.lower() == 'train':  # Если это Train
            train_tasks.append(task)
            # train_tasks_.append(name)
        elif subset.lower() == 'validation':  # Если это Val
            val_tasks.append(task)
            # val_tasks_.append(name)
        elif subset.lower() == 'test':  # Если это Test
            test_tasks.append(task)
            # test_tasks_.append(name)
        else:
            # Если подвыборка явно не классифицирована, заносим её в отдельный
            # словарь:
            other_tasks_dict[subset] = other_tasks_dict.get(subset, []) + [task]
            print(f'Неоднозначная метка "{subset}" задачи "{name}"',
                  f'в папке "{task_dir}"!')
    '''
    print(*sorted(train_tasks_), sep='\n', end='\n\n')
    print(*sorted(  val_tasks_), sep='\n', end='\n\n')
    print(*sorted( test_tasks_), sep='\n', end='\n\n')
    '''
    return train_tasks, val_tasks, test_tasks, other_tasks_dict


def flat_tasks(tasks):
    '''
    Разбивает задачи для более эффективного распараллеливания при сохранении.
    Разделение выборки на train/val/test должно произойти до!
    '''
    # Словарь файл -> Задача:
    files = {}

    # Перебор задач:
    for task in tasks:

        # Перебор подзадач:
        for subtask in task:

            # Получаем имя файла для текущей подзадачи:
            file = subtask[1]

            # Если файл уже встречался, то добавляем эту
            # задачу в соответствующий список в словаре:
            if file in files:
                files[file] += [subtask]

            # Если файл ещё не встречался, то добавляем
            # эту задачу в новый список в словаре:
            else:
                files[file] = [subtask]

    # Собираем и возвращаем новый список задач:
    return list(files.values())


def sort_tasks_by_len(tasks, *args):
    '''
    Сортирует список задач по убыванию числа входящих кадров.
    
    Параллельная обработка отсортированных таким образом 
    задач выполняется несколько эффективнее.
    '''
    # Формируем список чисел кадров по каждой задаче:
    task_lens = []
    for task in tasks:
        task_len = 0
        for df, file, true_frames in task:
            task_len += len(true_frames)
        
        task_lens.append(task_len)
    
    # Получаем отсортированный список индексов:
    sorted_inds = np.argsort(task_lens)[::-1]
    
    # Сортируем длины задач, сами задачи и другие списки, если они представлены:
    task_lens, tasks, *args = reorder_lists(sorted_inds, task_lens, tasks, *args)
    
    return task_lens, tasks, *args


def drop_bbox_labled_cvat_tasks(tasks):
    '''
    Оставляет только те задачи, что размечены не обрамляющими прямоугольниками.
    Тип датасета определяется по имени папки, в которой он хранится.
    Датасет, размеченный прямоугольниками содержит в своём имени следующую строку:
    'project_video_annotations_bounding boxes_backup'
    '''
    # Перебираем все задачи:
    tasks_ = []
    for task in tasks:
        
        # Переносим лишь те подзадачи, что не имеют "project_video_annotations_bounding boxes_backup" в пути к файлу:
        task_ = [subtask for subtask in task if 'project_video_annotations_bounding boxes_backup' not in subtask[1]]
        
        # Пустые задачи не вносим:
        if len(task_):
            tasks_.append(task_)
    
    return tasks_


def crop_df_labels(df, bbox, area_part_th):
    '''
    Вовзращает датафрейм с метками, сохранившимися после вырезания фрагмента из исходного изображения.
    '''
    # Результирующий датафрейм:
    cropped_df = df.copy()
    
    # Функция, возвращающая пересечение текущего сегмента с заданной рамкой:
    def crop_func(row):
        points0 = CVATPoints(row['points'], type_=row['type'], rotation=row['rotation']).asbbox()
        points1 = points0.crop(bbox)
        
        # Возвращаем пересечение текущего сегмента только если ...
        # ... отношение площади новой фигуры к площади старой ...
        # ... не меньше заданного порога:
        
        if points1 is None:
            return
        
        else:
            s0 = points0.area()
            s1 = points1.area()
            '''
            if not isinstance(s0, (float, int)):
                print(type(s0))
                raise
            if not isinstance(s1, (float, int)):
                print(type(s1))
                raise
            '''
            if s0 == 0 or s1 / s0 < area_part_th:
                return
            
            else:
                return points1.flatten()
    
    # Применение crop_func ко всем меткам датафрейма, если последний не пуст:
    if len(cropped_df):
        cropped_df['points'] = cropped_df[['points', 'type', 'rotation']].apply(crop_func, axis=1)
        cropped_df['type'] = 'rectangle'
    # При этом все метки конвертируются в обрамляющие прямоугольники.
    
    # Возвращаем строки, прямоугольники которых имели пересечение с рамкой:
    return cropped_df[cropped_df['points'].notna()]


def split_image_and_labels2tiles(df           : 'Исходный датафрейм с разметкой'                                  ,
                                 image        : 'Исходное изображение'                                            ,
                                 max_im_size  : 'Максимальный размер целевых изобаржений'           = (1080, 1920),
                                 area_part_th : 'Доля площади, при которой сегмент ещё сохраняется' = 0.5         ):
    '''
    Разрезает кадр и метки на части заданного рзмера (полученные части могут пересекаться).
    Полезно при приведении к заданному размеру слишком больших изображений. Даже если
    исходное изображение меньше целевого, перед возвращением исходных данных все метки
    вписываются в рамки исходного изображения (исправляется выход за границы в исходной разметке).
    '''
    # Получаем размеры исходного и целевого изображений в виде numpy-векторов для удобства работы:
    im_size     = np.array(image.shape[:2])
    max_im_size = np.array(max_im_size)
    
    # Возвращаем данные, если исходное изображение меньше целевого:
    if (max_im_size >= im_size).all():
        im_rect = [0, 0, im_size[1], im_size[0]]
        return [(crop_df_labels(df, im_rect, area_part_th), image)]
    # При этом обрезаем края всех меток, что вышли за границы исходного изображения.
    
    # Определяем число сдвигов рамки вырезания по каждой оси:
    nij = np.ceil(im_size / max_im_size).astype(int)
    
    # Определяем шаг сдвига рамки вырезания по каждой оси:
    step = [0 if nij[ind] == 1 else (im_size[ind] - max_im_size[ind]) / (nij[ind] - 1) for ind in range(2)]
    # Заменяем нулями все inf-ы, оставшиеся после деления на 0.
    
    #step = (im_size - max_im_size) / (nij - 1)
    #np.nan_to_num(step, False, 0, 0, 0)s
    
    # Целевой список:
    tiles = []
    
    # Перебираем все сдвиги по веркикали:
    for i in range(nij[0]):
        
        # Определяем вертикальные границы рамки:
        y_min = int(i * step[0])
        y_max = min(y_min + max_im_size[0], im_size[0])
        
        # Перебираем все сдвиги по горизонтали:
        for j in range(nij[1]):
            
            # Определяем горизонтальные границы рамки:
            x_min = int(j * step[1])
            x_max = min(x_min + max_im_size[1], im_size[1])
            
            # Вырезаем фрагмент из изображения и разметки:
            cropped_image = image[y_min:y_max, x_min:x_max, :]
            cropped_df    = crop_df_labels(df, [x_min, y_min, x_max, y_max], area_part_th)
            
            # Внесение новой пары разметка-изображение в целевой список:
            tiles.append((cropped_df, cropped_image))
    
    # Возвращаем целевой список:
    return tiles


def fill_na_in_track_id(df):
    '''
    Заполняет незаполненные идентификаторы объектов уникальными номерами.
    '''
    # Копируем датафрейм, чтобы не перезаписывать исходный:
    df = df.copy()

    # Получаем маску объектов с незаполненным полем 'track_id':
    None_in_track_id_mask = df['track_id'] != df['track_id']

    # Если пропуски вообще есть:
    if None_in_track_id_mask.any():

        # Если не задан ни один 'track_id', то заполняем все подряд:
        if None_in_track_id_mask.all():
            df['track_id'] = range(len(df))

        # Если есть как заданные, так и пропущенные значения поля 'track_id':
        else:
            # Получаем номер, с которого можно продолжить нумерацию
            # 'track_id':
            start_track_id = df[~None_in_track_id_mask]['track_id'].max()

            # Продолжаем нумерацию для незаполненных объектов:
            df.loc[None_in_track_id_mask, 'track_id'] = range(
                start_track_id,
                start_track_id + None_in_track_id_mask.sum()
            )

    return df


def fill_na_in_track_id_in_all_tasks(tasks):
    '''
    Заполняет незаполненные идентификаторы объектов уникальными номерами.
    '''
    # Инициализация конечного списка задач:
    tasks_ = []

    # Перебор всех задач:
    for task in tasks:

        # Меняем df в каждой подзадаче:
        task = [(fill_na_in_track_id(df), file, true_frames) for df, file, true_frames in task]

        # Обновлённую задачу вносим в конечный список:
        tasks_.append(task)

    return tasks_


def sort_tasks(tasks):
    '''
    Сортирует все задачи по имени файла первой подзадачи.
    Используется перед конвертацией для унификации результатов конвертации.
    '''
    # Список имён первых файлов для первой подзадачи каждой из задачь:
    files = []
    for task in tasks:

        # Имя файла первой подзадачи текущей задачи:
        file = task[0][1]

        # Если файлов несколько, берём первый:
        if isinstance(file, (list, tuple)):
            file = file[0]

        # Вносим в список для сортировки
        files.append(file)

    # Возвращаем отсортированный список задач:
    return [tasks[ind] for ind in np.argsort(files)]


def img_or_size2img_and_size(img_or_size):
    '''
    Определяет изображение и его размер:
    '''
    img_or_size = np.array(img_or_size)
    if img_or_size.size in {2, 3}:
        img = np.zeros(img_or_size, np.uint8)
        imsize = img_or_size
    else:
        img = img_or_size
        imsize = img.shape

    return img, imsize


def draw_df_frame(df_frame,
                  img_or_size,
                  label2color=None,
                  caption=False,
                  thickness=None):
    '''
    Последовательная отрисовка всех объектов из датафрейма на изображении.
    '''
    # Определяем изображение и его размер:
    img, imsize = img_or_size2img_and_size(img_or_size)

    # Доопределяем цвет отрисовки, если словарь не задан:
    if label2color is None:
        if len(imsize) == 2 or imsize[2] == 1:
            color = 255
        else:
            color = (255, 255, 255)

    # Сама отрисовка:
    for dfrow in df_frame.iloc:

        # Пропускаем скрытые объекты:
        if dfrow['outside']:
            continue

        points = CVATPoints.from_dfrow(dfrow, imsize=imsize)
        label = dfrow['label']  # Метка объекта
        if label2color is not None:
            color = label2color[label]  # Цвет отрисовки объекта
        img = points.draw(img,
                          caption=label if caption else None,
                          color=color,
                          thickness=thickness)

    return img


def ergonomic_draw_df_frame(df_frame,
                            img_or_size,
                            label2color=None,
                            alpha=0.5):
    '''
    Визуализация разметки одного кадра в более комфортном для восприятия
    формате.
    '''
    # Определяем изображение и его размер:
    img, imsize = img_or_size2img_and_size(img_or_size)

    # Залитые сегменты:
    labeled_img = draw_df_frame(df_frame,
                                img_or_size=img,
                                label2color=label2color,
                                caption=False,
                                thickness=-1)

    # Наложение залитых сегментов на исходное изображение:
    labeled_img = (labeled_img * (1. - alpha) + img * alpha).astype(img.dtype)

    # Добавление контуров сегментов:
    labeled_img = draw_df_frame(df_frame,
                                img_or_size=labeled_img,
                                label2color=label2color,
                                caption=False,
                                thickness=5)

    # Наненсение надписей:
    labeled_img = draw_df_frame(df_frame,
                                img_or_size=labeled_img,
                                label2color=None,
                                caption=True,
                                thickness=0)

    return labeled_img


def subtask2preview(subtask,
                    out_file='./preview.mp4',
                    label2color=None,
                    alpha=0.5,
                    fps=3,
                    postprocessor=None,
                    recompress2mp4=True,
                    desc=None):
    '''
    Создать видеофайл-превью подзадачи.
    '''
    # Если тип итогового файла не поддерживается OpenCV, то пересжатие
    # включаем принудительно:
    if os.path.splitext(out_file)[-1].lower() != '.avi':
        recompress2mp4 = True

    # Разбиваем подзадачу на составляющие:
    df, file, true_frames = subtask

    # Инициируем пустой датафрейм, если передан None:
    if df is None:
        df = new_df()

    # Создаём словарь цветов, если он не был указан:
    if label2color is None:

        # Создаём упорядоченный список используемых меток:
        labels = sorted(list(df['label'].unique()))

        # Заполняем словарь цветами по кругу, используюя
        # color_float_hsv_to_uint8_rgb:
        label2color = {}
        for ind, label in enumerate(labels):
            label2color[label] = color_float_hsv_to_uint8_rgb(
                ind / len(labels))

    # Интерполируем разметку:
    df = interpolate_df(df, true_frames)

    # Создаём читатель исходного видеофайла:
    imgs = VideoGenerator(file)

    # Путь ко временному файлу:
    tmp_file = out_file + '_tmp.avi' if recompress2mp4 else out_file

    # Пишем временный файл:
    with ViSave(tmp_file, fps=fps) as v:
        for true_frame in tqdm(true_frames.values(),
                               desc=desc,
                               disable=desc is None):

            # Наносим разметку на исходное изображение:
            frame_df = df[df['true_frame'] == true_frame]
            img = imgs[true_frame]
            img = ergonomic_draw_df_frame(frame_df, img, label2color, alpha)

            # Наносим номер кадра:
            img = draw_contrast_text(img, str(true_frame))

            # Выполняем постобработку, если надо:
            if postprocessor is not None:
                img = postprocessor(img)

            # Сохраняем очередной кадр во временный файл:
            v(img)

    # Сброс постобработчика, если возможно:
    if hasattr(postprocessor, 'reset'):
        postprocessor.reset()

    # Пересжатие временного файла в итоговый, если надо:
    if recompress2mp4:
        recomp2mp4(tmp_file, out_file)


def tasks2preview(tasks,
                  out_file='./preview.mp4',
                  label2color=None,
                  alpha=0.5,
                  fps=3,
                  postprocessor=None,
                  recompress2mp4=True,
                  desc='Формирование общего превью',
                  **kwargs):
    '''
    Формирование одного превью-видеофайла для всего списка задач.
    '''
    # Выносим каждую подзадачу в отдельную задачу:
    sorted_tasks = flat_tasks(tasks)
    # Нужно для ускорения распараллеливания.

    # Формируем список имён видеопревью для каждой задачи:
    name, ext = os.path.splitext(out_file)
    sorted_out_files = [name + f'_{ind:08}' + ext
                        for ind in range(len(tasks))]

    # Меняем очерёдность списка задач и файлов превью для ускорения обработки:
    _, tasks, out_files = sort_tasks_by_len(sorted_tasks, sorted_out_files)

    # Формируем список подзадач:
    subtasks = [task[0] for task in tasks]

    # Создаём превью отдельно для каждой поздадачи:
    mpmap(subtask2preview,
          subtasks,
          out_files,
          [label2color] * len(subtasks),
          [alpha] * len(subtasks),
          [fps] * len(subtasks),
          [postprocessor] * len(subtasks),
          [recompress2mp4] * len(subtasks),
          desc=desc, **kwargs)

    # Создаём файл-список видео для зборки:
    file_list = f'{out_file}.list'
    with open(file_list, 'w') as f:
        for sorted_out_file in sorted_out_files:
            print(sorted_out_file)
            f.write(f"file '{os.path.basename(sorted_out_file)}'\n")

    # Выполняем сборку без пересжатия:
    os.system(f'ffmpeg -y -f concat -safe 0  -i "{file_list}" -c copy "{out_file}"')

    # Удаляем файл-список и файлы-фрагменты:
    os.remove(file_list)
    for file in sorted_out_files:
        os.remove(file_list)

    return out_file


"""
# Код тестирования поворотов и отражений в CVATPoints:
for rotation in [0, 15]:
    for points in [CVATPoints([100, 200, 400, 350], 'rectangle', imsize=(500, 600), rotation=rotation, rotate_immediately=False),
                   CVATPoints([100, 200, 400, 350], 'ellipse'  , imsize=(500, 600), rotation=rotation, rotate_immediately=False),
                   CVATPoints([100, 200, 400, 350 ,
                               300, 150, 250, 120],              imsize=(500, 600), rotation=rotation, rotate_immediately=False)]:
        
        img = points.draw(show_start_point=True)
        
        gt_flip = np.hstack([              img ,
                                 np.fliplr(img),
                                 np.flipud(img),
                                 np.flip  (img, (0, 1))])
        
        my_flip = np.hstack([points.flip().flip().draw(show_start_point=True),
                             points.fliplr()     .draw(show_start_point=True),
                             points.flipud()     .draw(show_start_point=True),
                             points.flip  ()     .draw(show_start_point=True)])
        
        flip = np.dstack([gt_flip, my_flip, my_flip])
        
        gt_rot = np.hstack([img] * 4)
        
        my_rot = np.hstack([np.rot90(points.rot90(k).draw(show_start_point=True), -k) for k in range(4)])
        
        rot = np.dstack([gt_rot, my_rot, my_rot])
        
        plt.figure(figsize=(24, 24))
        plt.imshow(np.vstack([flip, rot]));
        plt.axis(False)
"""


'''
# Применяет метод ко всем подобъектам:
def _call_method2all_data(data: list,
                          method: str,
                          *args,
                          desc=None,
                          num_procs: int = 1) -> list:

    # Список методов для каждой функции:
    functions = [getattr(sub_data, method) for sub_data in data]

    # Список аргументов:
    len_ = len(data)
    args_ = [[arg] * len_ for arg in args]

    # Выполняем функции:
    return mpmap(exec_function, functions, *args_, )
'''


#__all__ = 'CVATPoints', 'cvat_backups2tasks', 'drop_bbox_labled_cvat_tasks', 'sort_tasks'